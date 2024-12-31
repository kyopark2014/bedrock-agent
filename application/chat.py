import traceback
import boto3
import os
import json
import re
import requests
import datetime
import functools
import uuid
import time
import logging
import base64

from io import BytesIO
from PIL import Image
from pytz import timezone
from langchain_aws import ChatBedrock
from botocore.config import Config
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.tools import tool
from langchain.docstore.document import Document
from tavily import TavilyClient  
from langchain_community.tools.tavily_search import TavilySearchResults
from bs4 import BeautifulSoup
from botocore.exceptions import ClientError
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.graph import START, END, StateGraph
from typing import Any, List, Tuple, Dict, Optional, cast, Literal, Sequence, Union
from typing_extensions import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
from langchain_aws import AmazonKnowledgeBasesRetriever
from multiprocessing import Process, Pipe
from urllib import parse
from pydantic.v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser

try:
    with open("/home/config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
        print('config: ', config)
except Exception:
    print("use local configuration")
    with open("application/config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
        print('config: ', config)

bedrock_region = "us-west-2"
projectName = config["projectName"] if "projectName" in config else "bedrock-agent"

accountId = config["accountId"] if "accountId" in config else None
if accountId is None:
    raise Exception ("No accountId")

region = config["region"] if "region" in config else "us-west-2"
print('region: ', region)

bucketName = config["bucketName"] if "bucketName" in config else f"storage-for-{projectName}-{accountId}-{region}" 
print('bucketName: ', bucketName)

s3_prefix = 'docs'

knowledge_base_role = config["knowledge_base_role"] if "knowledge_base_role" in config else None
if knowledge_base_role is None:
    raise Exception ("No Knowledge Base Role")

collectionArn = config["collectionArn"] if "collectionArn" in config else None
if collectionArn is None:
    raise Exception ("No collectionArn")

vectorIndexName = projectName

opensearch_url = config["opensearch_url"] if "opensearch_url" in config else None
if opensearch_url is None:
    raise Exception ("No OpenSearch URL")

credentials = boto3.Session().get_credentials()
service = "aoss" 
awsauth = AWSV4SignerAuth(credentials, region, service)

s3_arn = config["s3_arn"] if "s3_arn" in config else None
if s3_arn is None:
    raise Exception ("No S3 ARN")

parsingModelArn = f"arn:aws:bedrock:{region}::foundation-model/anthropic.claude-3-haiku-20240307-v1:0"
embeddingModelArn = f"arn:aws:bedrock:{region}::foundation-model/amazon.titan-embed-text-v2:0"

knowledge_base_name = projectName

prompt_flow_name = 'aws-bot'
rag_prompt_flow_name = 'rag-prompt-flow'
knowledge_base_name = 'aws-rag'

numberOfDocs = 4
MSG_LENGTH = 100    
grade_state = "LLM" # LLM, PRIORITY_SEARCH, OTHERS
multi_region = 'disable'
minDocSimilarity = 400
length_of_models = 1
doc_prefix = s3_prefix+'/'
path = 'https://url/'   

os_client = OpenSearch(
    hosts = [{
        'host': opensearch_url.replace("https://", ""), 
        'port': 443
    }],
    http_auth=awsauth,
    use_ssl = True,
    verify_certs = True,
    connection_class=RequestsHttpConnection,
)

multi_region_models = [   # Nova Pro
    {   
        "bedrock_region": "us-west-2", # Oregon
        "model_type": "nova",
        "model_id": "us.amazon.nova-pro-v1:0"
    },
    {
        "bedrock_region": "us-east-1", # N.Virginia
        "model_type": "nova",
        "model_id": "us.amazon.nova-pro-v1:0"
    },
    {
        "bedrock_region": "us-east-2", # Ohio
        "model_type": "nova",
        "model_id": "us.amazon.nova-pro-v1:0"
    }
]
selected_chat = 0
HUMAN_PROMPT = "\n\nHuman:"
AI_PROMPT = "\n\nAssistant:"

userId = "demo"
map_chain = dict() 

def is_not_exist(index_name):    
    print('index_name: ', index_name)
        
    if os_client.indices.exists(index_name):
        print('use exist index: ', index_name)    
        return False
    else:
        print('no index: ', index_name)
        return True
    
knowledge_base_id = ""
data_source_id = ""
def initiate_knowledge_base():
    global knowledge_base_id, data_source_id
    #########################
    # opensearch index
    #########################
    if(is_not_exist(vectorIndexName)):
        print(f"creating opensearch index... {vectorIndexName}")        
        body={ 
            'settings':{
                "index.knn": True,
                "index.knn.algo_param.ef_search": 512,
                'analysis': {
                    'analyzer': {
                        'my_analyzer': {
                            'char_filter': ['html_strip'], 
                            'tokenizer': 'nori',
                            'filter': ['nori_number','lowercase','trim','my_nori_part_of_speech'],
                            'type': 'custom'
                        }
                    },
                    'tokenizer': {
                        'nori': {
                            'decompound_mode': 'mixed',
                            'discard_punctuation': 'true',
                            'type': 'nori_tokenizer'
                        }
                    },
                    "filter": {
                        "my_nori_part_of_speech": {
                            "type": "nori_part_of_speech",
                            "stoptags": [
                                    "E", "IC", "J", "MAG", "MAJ",
                                    "MM", "SP", "SSC", "SSO", "SC",
                                    "SE", "XPN", "XSA", "XSN", "XSV",
                                    "UNA", "NA", "VSV"
                            ]
                        }
                    }
                },
            },
            'mappings': {
                'properties': {
                    'vector_field': {
                        'type': 'knn_vector',
                        'dimension': 1024,
                        'method': {
                            "name": "hnsw",
                            "engine": "faiss",
                            "parameters": {
                                "ef_construction": 512,
                                "m": 16
                            }
                        }                  
                    },
                    "AMAZON_BEDROCK_METADATA": {"type": "text", "index": False},
                    "AMAZON_BEDROCK_TEXT_CHUNK": {"type": "text"},
                }
            }
        }

        try: # create index
            response = os_client.indices.create(
                vectorIndexName,
                body=body
            )
            print('opensearch index was created:', response)

            # delay 3seconds
            time.sleep(5)
        except Exception:
            err_msg = traceback.format_exc()
            print('error message: ', err_msg)                
            #raise Exception ("Not able to create the index")
            
    #########################
    # knowledge base
    #########################
    print('knowledge_base_name: ', knowledge_base_name)
    print('collectionArn: ', collectionArn)
    print('vectorIndexName: ', vectorIndexName)
    print('embeddingModelArn: ', embeddingModelArn)
    print('knowledge_base_role: ', knowledge_base_role)
    try: 
        client = boto3.client(
            service_name='bedrock-agent',
            region_name=bedrock_region
        )   

        response = client.list_knowledge_bases(
            maxResults=10
        )
        print('(list_knowledge_bases) response: ', response)
        
        if "knowledgeBaseSummaries" in response:
            summaries = response["knowledgeBaseSummaries"]
            for summary in summaries:
                if summary["name"] == knowledge_base_name:
                    knowledge_base_id = summary["knowledgeBaseId"]
                    print('knowledge_base_id: ', knowledge_base_id)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)
                    
    if not knowledge_base_id:
        print('creating knowledge base...')        
        for atempt in range(3):
            try:
                response = client.create_knowledge_base(
                    name=knowledge_base_name,
                    description="Knowledge base based on OpenSearch",
                    roleArn=knowledge_base_role,
                    knowledgeBaseConfiguration={
                        'type': 'VECTOR',
                        'vectorKnowledgeBaseConfiguration': {
                            'embeddingModelArn': embeddingModelArn,
                            'embeddingModelConfiguration': {
                                'bedrockEmbeddingModelConfiguration': {
                                    'dimensions': 1024
                                }
                            }
                        }
                    },
                    storageConfiguration={
                        'type': 'OPENSEARCH_SERVERLESS',
                        'opensearchServerlessConfiguration': {
                            'collectionArn': collectionArn,
                            'fieldMapping': {
                                'metadataField': 'AMAZON_BEDROCK_METADATA',
                                'textField': 'AMAZON_BEDROCK_TEXT_CHUNK',
                                'vectorField': 'vector_field'
                            },
                            'vectorIndexName': vectorIndexName
                        }
                    }                
                )   
                print('(create_knowledge_base) response: ', response)
            
                if 'knowledgeBaseId' in response['knowledgeBase']:
                    knowledge_base_id = response['knowledgeBase']['knowledgeBaseId']
                    break
                else:
                    knowledge_base_id = ""    
            except Exception:
                    err_msg = traceback.format_exc()
                    print('error message: ', err_msg)
                    time.sleep(5)
                    print(f"retrying... ({atempt})")
                    #raise Exception ("Not able to create the knowledge base")      
                
    print(f"knowledge_base_name: {knowledge_base_name}, knowledge_base_id: {knowledge_base_id}")    
    
    #########################
    # data source      
    #########################
    data_source_name = bucketName  
    try: 
        response = client.list_data_sources(
            knowledgeBaseId=knowledge_base_id,
            maxResults=10
        )        
        print('(list_data_sources) response: ', response)
        
        if 'dataSourceSummaries' in response:
            for data_source in response['dataSourceSummaries']:
                print('data_source: ', data_source)
                if data_source['name'] == data_source_name:
                    data_source_id = data_source['dataSourceId']
                    print('data_source_id: ', data_source_id)
                    break    
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)
        
    if not data_source_id:
        print('creating data source...')  
        try:
            response = client.create_data_source(
                dataDeletionPolicy='DELETE',
                dataSourceConfiguration={
                    's3Configuration': {
                        'bucketArn': s3_arn,
                        'inclusionPrefixes': [ 
                            s3_prefix+'/',
                        ]
                    },
                    'type': 'S3'
                },
                description = f"S3 data source: {bucketName}",
                knowledgeBaseId = knowledge_base_id,
                name = data_source_name,
                vectorIngestionConfiguration={
                    'chunkingConfiguration': {
                        'chunkingStrategy': 'HIERARCHICAL',
                        'hierarchicalChunkingConfiguration': {
                            'levelConfigurations': [
                                {
                                    'maxTokens': 1500
                                },
                                {
                                    'maxTokens': 300
                                }
                            ],
                            'overlapTokens': 60
                        }
                    },
                    'parsingConfiguration': {
                        'bedrockFoundationModelConfiguration': {
                            'modelArn': parsingModelArn
                        },
                        'parsingStrategy': 'BEDROCK_FOUNDATION_MODEL'
                    }
                }
            )
            print('(create_data_source) response: ', response)
            
            if 'dataSource' in response:
                if 'dataSourceId' in response['dataSource']:
                    data_source_id = response['dataSource']['dataSourceId']
                    print('data_source_id: ', data_source_id)
                    
        except Exception:
            err_msg = traceback.format_exc()
            print('error message: ', err_msg)
            #raise Exception ("Not able to create the data source")
    
    print(f"data_source_name: {data_source_name}, data_source_id: {data_source_id}")
            
initiate_knowledge_base()

if userId in map_chain:  
        # print('memory exist. reuse it!')
        memory_chain = map_chain[userId]
else: 
    # print('memory does not exist. create new one!')        
    memory_chain = ConversationBufferWindowMemory(memory_key="chat_history", output_key='answer', return_messages=True, k=5)
    map_chain[userId] = memory_chain

def get_chat():
    global selected_chat
    
    profile = multi_region_models[selected_chat]
    length_of_models = len(multi_region_models)
        
    bedrock_region =  profile['bedrock_region']
    modelId = profile['model_id']
    maxOutputTokens = 4096
    print(f'LLM: {selected_chat}, bedrock_region: {bedrock_region}, modelId: {modelId}')
                          
    # bedrock   
    boto3_bedrock = boto3.client(
        service_name='bedrock-runtime',
        region_name=bedrock_region,
        config=Config(
            retries = {
                'max_attempts': 30
            }
        )
    )
    parameters = {
        "max_tokens":maxOutputTokens,     
        "temperature":0.1,
        "top_k":250,
        "top_p":0.9,
        "stop_sequences": [HUMAN_PROMPT]
    }
    # print('parameters: ', parameters)

    chat = ChatBedrock(   # new chat model
        model_id=modelId,
        client=boto3_bedrock, 
        model_kwargs=parameters,
        region_name=bedrock_region
    )    
    
    selected_chat = selected_chat + 1
    if selected_chat == length_of_models:
        selected_chat = 0
    
    return chat

def get_multi_region_chat(models, selected):
    profile = models[selected]
    bedrock_region =  profile['bedrock_region']
    modelId = profile['model_id']
    maxOutputTokens = 4096
    print(f'selected_chat: {selected}, bedrock_region: {bedrock_region}, modelId: {modelId}')
                          
    # bedrock   
    boto3_bedrock = boto3.client(
        service_name='bedrock-runtime',
        region_name=bedrock_region,
        config=Config(
            retries = {
                'max_attempts': 30
            }
        )
    )
    parameters = {
        "max_tokens":maxOutputTokens,     
        "temperature":0.1,
        "top_k":250,
        "top_p":0.9,
        "stop_sequences": [HUMAN_PROMPT]
    }
    # print('parameters: ', parameters)

    chat = ChatBedrock(   # new chat model
        model_id=modelId,
        client=boto3_bedrock, 
        model_kwargs=parameters,
    )        
    return chat

def general_conversation(query):
    chat = get_chat()

    system = (
        "당신의 이름은 서연이고, 질문에 대해 친절하게 답변하는 사려깊은 인공지능 도우미입니다."
        "상황에 맞는 구체적인 세부 정보를 충분히 제공합니다." 
        "모르는 질문을 받으면 솔직히 모른다고 말합니다."
        "답변은 markdown 포맷(예: ##)을 사용하지 않습니다."
    )
    
    human = "Question: {input}"
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system), 
        MessagesPlaceholder(variable_name="history"), 
        ("human", human)
    ])
                
    history = memory_chain.load_memory_variables({})["chat_history"]

    chain = prompt | chat | StrOutputParser()
    try: 
        stream = chain.stream(
            {
                "history": history,
                "input": query,
            }
        )  
        print('stream: ', stream)
            
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)        
        raise Exception ("Not able to request to LLM: "+err_msg)
        
    return stream

def print_doc(i, doc):
    if len(doc.page_content)>=100:
        text = doc.page_content[:100]
    else:
        text = doc.page_content
            
    print(f"{i}: {text}, metadata:{doc.metadata}")

def get_docs_from_knowledge_base(documents):        
    relevant_docs = []
    for doc in documents:
        content = ""
        if doc.page_content:
            content = doc.page_content
        
        score = doc.metadata["score"]
        
        link = ""
        if "s3Location" in doc.metadata["location"]:
            link = doc.metadata["location"]["s3Location"]["uri"] if doc.metadata["location"]["s3Location"]["uri"] is not None else ""
            
            # print('link:', link)    
            pos = link.find(f"/{doc_prefix}")
            name = link[pos+len(doc_prefix)+1:]
            encoded_name = parse.quote(name)
            # print('name:', name)
            link = f"{path}{doc_prefix}{encoded_name}"
            
        elif "webLocation" in doc.metadata["location"]:
            link = doc.metadata["location"]["webLocation"]["url"] if doc.metadata["location"]["webLocation"]["url"] is not None else ""
            name = "WEB"

        url = link
        # print('url:', url)
        
        relevant_docs.append(
            Document(
                page_content=content,
                metadata={
                    'name': name,
                    'score': score,
                    'url': url,
                    'from': 'RAG'
                },
            )
        )    
    return relevant_docs

def grade_document_based_on_relevance(conn, question, doc, models, selected):     
    chat = get_multi_region_chat(models, selected)
    retrieval_grader = get_retrieval_grader(chat)
    score = retrieval_grader.invoke({"question": question, "document": doc.page_content})
    # print(f"score: {score}")
    
    grade = score.binary_score    
    if grade == 'yes':
        print("---GRADE: DOCUMENT RELEVANT---")
        conn.send(doc)
    else:  # no
        print("---GRADE: DOCUMENT NOT RELEVANT---")
        conn.send(None)
    
    conn.close()

def grade_documents_using_parallel_processing(question, documents):
    global selected_chat
    
    filtered_docs = []    

    processes = []
    parent_connections = []
    
    for i, doc in enumerate(documents):
        #print(f"grading doc[{i}]: {doc.page_content}")        
        parent_conn, child_conn = Pipe()
        parent_connections.append(parent_conn)
            
        process = Process(target=grade_document_based_on_relevance, args=(child_conn, question, doc, multi_region_models, selected_chat))
        processes.append(process)

        selected_chat = selected_chat + 1
        if selected_chat == length_of_models:
            selected_chat = 0
    for process in processes:
        process.start()
            
    for parent_conn in parent_connections:
        relevant_doc = parent_conn.recv()

        if relevant_doc is not None:
            filtered_docs.append(relevant_doc)

    for process in processes:
        process.join()
    
    #print('filtered_docs: ', filtered_docs)
    return filtered_docs

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

def get_retrieval_grader(chat):
    system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )
    
    structured_llm_grader = chat.with_structured_output(GradeDocuments)
    retrieval_grader = grade_prompt | structured_llm_grader
    return retrieval_grader

def grade_documents(question, documents):
    print("###### grade_documents ######")
    
    print("start grading...")
    print("grade_state: ", grade_state)
    
    if grade_state == "LLM":
        filtered_docs = []
        if multi_region == 'enable':  # parallel processing        
            filtered_docs = grade_documents_using_parallel_processing(question, documents)

        else:
            # Score each doc    
            chat = get_chat()
            retrieval_grader = get_retrieval_grader(chat)
            for i, doc in enumerate(documents):
                # print('doc: ', doc)
                print_doc(i, doc)
                
                score = retrieval_grader.invoke({"question": question, "document": doc.page_content})
                # print("score: ", score)
                
                grade = score.binary_score
                # print("grade: ", grade)
                # Document relevant
                if grade.lower() == "yes":
                    print("---GRADE: DOCUMENT RELEVANT---")
                    filtered_docs.append(doc)
                # Document not relevant
                else:
                    print("---GRADE: DOCUMENT NOT RELEVANT---")
                    # We do not include the document in filtered_docs
                    # We set a flag to indicate that we want to run web search
                    continue
    
    else:  # OTHERS
        filtered_docs = documents

    return filtered_docs

contentList = []
def check_duplication(docs):
    global contentList
    length_original = len(docs)
    
    updated_docs = []
    print('length of relevant_docs:', len(docs))
    for doc in docs:            
        # print('excerpt: ', doc['metadata']['excerpt'])
            if doc.page_content in contentList:
                print('duplicated!')
                continue
            contentList.append(doc.page_content)
            updated_docs.append(doc)            
    length_updateed_docs = len(updated_docs)     
    
    if length_original == length_updateed_docs:
        print('no duplication')
    
    return updated_docs

def query_using_RAG_context(chat, context, revised_question):    
    if isKorean(revised_question)==True:
        system = (
            """다음의 <context> tag안의 참고자료를 이용하여 상황에 맞는 구체적인 세부 정보를 충분히 제공합니다. Assistant의 이름은 서연이고, 모르는 질문을 받으면 솔직히 모른다고 말합니다.
            
            <context>
            {context}
            </context>"""
        )
    else: 
        system = (
            """Here is pieces of context, contained in <context> tags. Provide a concise answer to the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
            
            <context>
            {context}
            </context>"""
        )
    
    human = "{input}"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    # print('prompt: ', prompt)
                   
    chain = prompt | chat
    
    try: 
        output = chain.invoke(
            {
                "context": context,
                "input": revised_question,
            }
        )
        msg = output.content
        print('msg: ', msg)
        
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)    
        raise Exception ("Not able to request to LLM: "+err_msg)

    return msg

def get_answer_using_knowledge_base(text):
    global reference_docs

    chat = get_chat()
    
    msg = reference = ""
    top_k = numberOfDocs
    relevant_docs = []
    if knowledge_base_id:    
        retriever = AmazonKnowledgeBasesRetriever(
            knowledge_base_id=knowledge_base_id, 
            retrieval_config={"vectorSearchConfiguration": {
                "numberOfResults": top_k,
                "overrideSearchType": "HYBRID"   # SEMANTIC
            }},
        )
        
        docs = retriever.invoke(text)
        # print('docs: ', docs)
        print('--> docs from knowledge base')
        for i, doc in enumerate(docs):
            print_doc(i, doc)
        
        relevant_docs = get_docs_from_knowledge_base(docs)
        
    # grading   
    filtered_docs = grade_documents(text, relevant_docs)    
    
    filtered_docs = check_duplication(filtered_docs) # duplication checker
            
    relevant_context = ""
    for i, document in enumerate(filtered_docs):
        print(f"{i}: {document}")
        if document.page_content:
            content = document.page_content
            
        relevant_context = relevant_context + content + "\n\n"
        
    print('relevant_context: ', relevant_context)

    msg = query_using_RAG_context(chat, relevant_context, text)
    
    if len(filtered_docs):
        reference_docs += filtered_docs 
            
    return msg

def run_flow(text):
    global reference_docs

    chat = get_chat()
    
    msg = reference = ""
    top_k = numberOfDocs
    relevant_docs = []
    if knowledge_base_id:    
        retriever = AmazonKnowledgeBasesRetriever(
            knowledge_base_id=knowledge_base_id, 
            retrieval_config={"vectorSearchConfiguration": {
                "numberOfResults": top_k,
                "overrideSearchType": "HYBRID"   # SEMANTIC
            }},
        )
        
        docs = retriever.invoke(text)
        # print('docs: ', docs)
        print('--> docs from knowledge base')
        for i, doc in enumerate(docs):
            print_doc(i, doc)
        
        relevant_docs = get_docs_from_knowledge_base(docs)
        
    # grading   
    filtered_docs = grade_documents(text, relevant_docs)    
    
    filtered_docs = check_duplication(filtered_docs) # duplication checker
            
    relevant_context = ""
    for i, document in enumerate(filtered_docs):
        print(f"{i}: {document}")
        if document.page_content:
            content = document.page_content
            
        relevant_context = relevant_context + content + "\n\n"
        
    print('relevant_context: ', relevant_context)

    msg = query_using_RAG_context(chat, relevant_context, text)
    
    if len(filtered_docs):
        reference_docs += filtered_docs 
            
    return msg

def run_bedrock_agent(text):
    global reference_docs

    chat = get_chat()
    
    msg = reference = ""
    top_k = numberOfDocs
    relevant_docs = []
    if knowledge_base_id:    
        retriever = AmazonKnowledgeBasesRetriever(
            knowledge_base_id=knowledge_base_id, 
            retrieval_config={"vectorSearchConfiguration": {
                "numberOfResults": top_k,
                "overrideSearchType": "HYBRID"   # SEMANTIC
            }},
        )
        
        docs = retriever.invoke(text)
        # print('docs: ', docs)
        print('--> docs from knowledge base')
        for i, doc in enumerate(docs):
            print_doc(i, doc)
        
        relevant_docs = get_docs_from_knowledge_base(docs)
        
    # grading   
    filtered_docs = grade_documents(text, relevant_docs)    
    
    filtered_docs = check_duplication(filtered_docs) # duplication checker
            
    relevant_context = ""
    for i, document in enumerate(filtered_docs):
        print(f"{i}: {document}")
        if document.page_content:
            content = document.page_content
            
        relevant_context = relevant_context + content + "\n\n"
        
    print('relevant_context: ', relevant_context)

    msg = query_using_RAG_context(chat, relevant_context, text)
    
    if len(filtered_docs):
        reference_docs += filtered_docs 
            
    return msg

def save_chat_history(text, msg):
    memory_chain.chat_memory.add_user_message(text)
    if len(msg) > MSG_LENGTH:
        memory_chain.chat_memory.add_ai_message(msg[:MSG_LENGTH])                          
    else:
        memory_chain.chat_memory.add_ai_message(msg) 

@tool 
def get_book_list(keyword: str) -> str:
    """
    Search book list by keyword and then return book list
    keyword: search keyword
    return: book list
    """
    
    keyword = keyword.replace('\'','')

    answer = ""
    url = f"https://search.kyobobook.co.kr/search?keyword={keyword}&gbCode=TOT&target=total"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        prod_info = soup.find_all("a", attrs={"class": "prod_info"})
        
        if len(prod_info):
            answer = "추천 도서는 아래와 같습니다.\n"
            
        for prod in prod_info[:5]:
            title = prod.text.strip().replace("\n", "")       
            link = prod.get("href")
            answer = answer + f"{title}, URL: {link}\n\n"
    
    return answer

@tool
def get_current_time(format: str=f"%Y-%m-%d %H:%M:%S")->str:
    """Returns the current date and time in the specified format"""
    # f"%Y-%m-%d %H:%M:%S"
    
    format = format.replace('\'','')
    timestr = datetime.datetime.now(timezone('Asia/Seoul')).strftime(format)
    print('timestr:', timestr)
    
    return timestr

@tool
def get_weather_info(city: str) -> str:
    """
    retrieve weather information by city name and then return weather statement.
    city: the name of city to retrieve
    return: weather statement
    """    
    
    city = city.replace('\n','')
    city = city.replace('\'','')
    city = city.replace('\"','')
                
    chat = get_chat()
    if isKorean(city):
        place = traslation(chat, city, "Korean", "English")
        print('city (translated): ', place)
    else:
        place = city
        city = traslation(chat, city, "English", "Korean")
        print('city (translated): ', city)
        
    print('place: ', place)
    
    weather_str: str = f"{city}에 대한 날씨 정보가 없습니다."
    if weather_api_key: 
        apiKey = weather_api_key
        lang = 'en' 
        units = 'metric' 
        api = f"https://api.openweathermap.org/data/2.5/weather?q={place}&APPID={apiKey}&lang={lang}&units={units}"
        # print('api: ', api)
                
        try:
            result = requests.get(api)
            result = json.loads(result.text)
            print('result: ', result)
        
            if 'weather' in result:
                overall = result['weather'][0]['main']
                current_temp = result['main']['temp']
                min_temp = result['main']['temp_min']
                max_temp = result['main']['temp_max']
                humidity = result['main']['humidity']
                wind_speed = result['wind']['speed']
                cloud = result['clouds']['all']
                
                weather_str = f"{city}의 현재 날씨의 특징은 {overall}이며, 현재 온도는 {current_temp}도 이고, 최저온도는 {min_temp}도, 최고 온도는 {max_temp}도 입니다. 현재 습도는 {humidity}% 이고, 바람은 초당 {wind_speed} 미터 입니다. 구름은 {cloud}% 입니다."
                #weather_str = f"Today, the overall of {city} is {overall}, current temperature is {current_temp} degree, min temperature is {min_temp} degree, highest temperature is {max_temp} degree. huminity is {humidity}%, wind status is {wind_speed} meter per second. the amount of cloud is {cloud}%."            
        except Exception:
            err_msg = traceback.format_exc()
            print('error message: ', err_msg)   
            # raise Exception ("Not able to request to LLM")    
        
    print('weather_str: ', weather_str)                            
    return weather_str

@tool
def search_by_tavily(keyword: str) -> str:
    """
    Search general information by keyword and then return the result as a string.
    keyword: search keyword
    return: the information of keyword
    """    
    global reference_docs    
    answer = ""
    
    if tavily_api_key:
        keyword = keyword.replace('\'','')
        
        search = TavilySearchResults(
            max_results=3,
            include_answer=True,
            include_raw_content=True,
            search_depth="advanced", # "basic"
            include_domains=["google.com", "naver.com"]
        )
                    
        try: 
            output = search.invoke(keyword)
            print('tavily output: ', output)
            
            for result in output:
                print('result: ', result)
                if result:
                    content = result.get("content")
                    url = result.get("url")
                    
                    reference_docs.append(
                        Document(
                            page_content=content,
                            metadata={
                                'name': 'WWW',
                                'url': url,
                                'from': 'tavily'
                            },
                        )
                    )                
                    answer = answer + f"{content}, URL: {url}\n"
        
        except Exception:
            err_msg = traceback.format_exc()
            print('error message: ', err_msg)           
            # raise Exception ("Not able to request to tavily")   
        
    return answer

tools = [get_current_time, get_book_list, get_weather_info, search_by_tavily]        

reference_docs = []
# api key to get weather information in agent
secretsmanager = boto3.client(
    service_name='secretsmanager',
    region_name=bedrock_region
)
try:
    get_weather_api_secret = secretsmanager.get_secret_value(
        SecretId=f"openweathermap-{projectName}"
    )
    #print('get_weather_api_secret: ', get_weather_api_secret)
    secret = json.loads(get_weather_api_secret['SecretString'])
    #print('secret: ', secret)
    weather_api_key = secret['weather_api_key']

except Exception as e:
    raise e

# api key to use LangSmith
langsmith_api_key = ""
try:
    get_langsmith_api_secret = secretsmanager.get_secret_value(
        SecretId=f"langsmithapikey-{projectName}"
    )
    #print('get_langsmith_api_secret: ', get_langsmith_api_secret)
    secret = json.loads(get_langsmith_api_secret['SecretString'])
    #print('secret: ', secret)
    langsmith_api_key = secret['langsmith_api_key']
    langchain_project = secret['langchain_project']
except Exception as e:
    raise e

if langsmith_api_key:
    os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = langchain_project

# api key to use Tavily Search
tavily_api_key = []
tavily_key = ""
try:
    get_tavily_api_secret = secretsmanager.get_secret_value(
        SecretId=f"tavilyapikey-{projectName}"
    )
    #print('get_tavily_api_secret: ', get_tavily_api_secret)
    secret = json.loads(get_tavily_api_secret['SecretString'])
    #print('secret: ', secret)
    tavily_key = secret['tavily_api_key']
    #print('tavily_api_key: ', tavily_api_key)
except Exception as e: 
    raise e

if tavily_key:
    os.environ["TAVILY_API_KEY"] = tavily_key

def tavily_search(query, k):
    docs = []    
    try:
        tavily_client = TavilyClient(api_key=tavily_key)
        response = tavily_client.search(query, max_results=k)
        # print('tavily response: ', response)
            
        for r in response["results"]:
            name = r.get("title")
            if name is None:
                name = 'WWW'
            
            docs.append(
                Document(
                    page_content=r.get("content"),
                    metadata={
                        'name': name,
                        'url': r.get("url"),
                        'from': 'tavily'
                    },
                )
            )                   
    except Exception as e:
        print('Exception: ', e)

    return docs

def isKorean(text):
    # check korean
    pattern_hangul = re.compile('[\u3131-\u3163\uac00-\ud7a3]+')
    word_kor = pattern_hangul.search(str(text))
    # print('word_kor: ', word_kor)

    if word_kor and word_kor != 'None':
        print('Korean: ', word_kor)
        return True
    else:
        print('Not Korean: ', word_kor)
        return False
    
def traslation(chat, text, input_language, output_language):
    system = (
        "You are a helpful assistant that translates {input_language} to {output_language} in <article> tags." 
        "Put it in <result> tags."
    )
    human = "<article>{text}</article>"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    # print('prompt: ', prompt)
    
    chain = prompt | chat    
    try: 
        result = chain.invoke(
            {
                "input_language": input_language,
                "output_language": output_language,
                "text": text,
            }
        )
        
        msg = result.content
        # print('translated text: ', msg)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)          
        raise Exception ("Not able to request to LLM")

    return msg[msg.find('<result>')+8:len(msg)-9] # remove <result> tag

####################### LangGraph #######################
# Chat Agent Executor 
# Agentic Workflow: Tool Use
#########################################################
def run_agent_executor2(query):        
    class State(TypedDict):
        messages: Annotated[list, add_messages]
        answer: str

    tool_node = ToolNode(tools)
            
    def create_agent(chat, tools):        
        tool_names = ", ".join([tool.name for tool in tools])
        print("tool_names: ", tool_names)

        system = (
            "당신의 이름은 서연이고, 질문에 친근한 방식으로 대답하도록 설계된 대화형 AI입니다."
            "상황에 맞는 구체적인 세부 정보를 충분히 제공합니다."
            "모르는 질문을 받으면 솔직히 모른다고 말합니다."

            "Use the provided tools to progress towards answering the question."
            "If you are unable to fully answer, that's OK, another assistant with different tools "
            "will help where you left off. Execute what you can to make progress."
            "You have access to the following tools: {tool_names}."
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system",system),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        
        prompt = prompt.partial(tool_names=tool_names)
        
        return prompt | chat.bind_tools(tools)
    
    def agent_node(state, agent, name):
        print(f"###### agent_node:{name} ######")

        last_message = state["messages"][-1]
        print('last_message: ', last_message)
        if isinstance(last_message, ToolMessage) and last_message.content=="":    
            print('last_message is empty') 
            answer = get_basic_answer(state["messages"][0].content)  
            return {
                "messages": [AIMessage(content=answer)],
                "answer": answer
            }
        
        response = agent.invoke(state["messages"])
        print('response: ', response)

        if "answer" in state:
            answer = state['answer']
        else:
            answer = ""

        for re in response.content:
            if "type" in re:
                if re['type'] == 'text':
                    print(f"--> {re['type']}: {re['text']}")
                elif re['type'] == 'tool_use':                
                    print(f"--> {re['type']}: name: {re['name']}, input: {re['input']}")
                else:
                    print(re)
            else: # answer
                answer += '\n'+response.content
                print(response.content)
                break

        response = AIMessage(**response.dict(exclude={"type", "name"}), name=name)     
        print('message: ', response)
        
        return {
            "messages": [response],
            "answer": answer
        }
    
    def final_answer(state):
        print(f"###### final_answer ######")        

        answer = ""        
        if "answer" in state:
            answer = state['answer']            
        else:
            answer = state["messages"][-1].content

        if answer.find('<thinking>') != -1:
            print('Remove <thinking> tag.')
            answer = answer[answer.find('</thinking>')+12:]
        print('answer: ', answer)
        
        return {
            "answer": answer
        }
    
    chat = get_chat()
    
    execution_agent = create_agent(chat, tools)
    
    execution_agent_node = functools.partial(agent_node, agent=execution_agent, name="execution_agent")
    
    def should_continue(state: State, config) -> Literal["continue", "end"]:
        print("###### should_continue ######")
        messages = state["messages"]    
        # print('(should_continue) messages: ', messages)
        
        last_message = messages[-1]        
        if not last_message.tool_calls:
            print("Final: ", last_message.content)
            print("--- END ---")
            return "end"
        else:      
            print(f"tool_calls: ", last_message.tool_calls)

            for message in last_message.tool_calls:
                print(f"tool name: {message['name']}, args: {message['args']}")
                # update_state_message(f"calling... {message['name']}", config)

            print(f"--- CONTINUE: {last_message.tool_calls[-1]['name']} ---")
            return "continue"

    def buildAgentExecutor():
        workflow = StateGraph(State)

        workflow.add_node("agent", execution_agent_node)
        workflow.add_node("action", tool_node)
        workflow.add_node("final_answer", final_answer)
        
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "continue": "action",
                "end": "final_answer",
            },
        )
        workflow.add_edge("action", "agent")
        workflow.add_edge("final_answer", END)

        return workflow.compile()

    app = buildAgentExecutor()
            
    inputs = [HumanMessage(content=query)]
    config = {"recursion_limit": 50}
    
    msg = ""
    # for event in app.stream({"messages": inputs}, config, stream_mode="values"):   
    #     # print('event: ', event)
        
    #     if "answer" in event:
    #         msg = event["answer"]
    #     else:
    #         msg = event["messages"][-1].content
    #     # print('message: ', message)

    output = app.invoke({"messages": inputs}, config)
    print('output: ', output)

    msg = output['answer']

    return msg

def get_basic_answer(query):
    print('#### get_basic_answer ####')
    chat = get_chat()

    if isKorean(query)==True:
        system = (
            "당신의 이름은 서연이고, 질문에 대해 친절하게 답변하는 사려깊은 인공지능 도우미입니다."
            "상황에 맞는 구체적인 세부 정보를 충분히 제공합니다." 
            "모르는 질문을 받으면 솔직히 모른다고 말합니다."
        )
    else: 
        system = (
            "You will be acting as a thoughtful advisor."
            "Using the following conversation, answer friendly for the newest question." 
            "If you don't know the answer, just say that you don't know, don't try to make up an answer."     
        )    
    
    human = "Question: {input}"    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system), 
        ("human", human)
    ])    
    
    chain = prompt | chat    
    output = chain.invoke({"input": query})
    print('output.content: ', output.content)

    return output.content

def translate_text(text):
    chat = get_chat()

    system = (
        "You are a helpful assistant that translates {input_language} to {output_language} in <article> tags. Put it in <result> tags."
    )
    human = "<article>{text}</article>"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    # print('prompt: ', prompt)
    
    if isKorean(text)==False :
        input_language = "English"
        output_language = "Korean"
    else:
        input_language = "Korean"
        output_language = "English"
                        
    chain = prompt | chat    
    try: 
        result = chain.invoke(
            {
                "input_language": input_language,
                "output_language": output_language,
                "text": text,
            }
        )
        msg = result.content
        print('translated text: ', msg)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)       
        raise Exception ("Not able to request to LLM")

    return msg[msg.find('<result>')+8:len(msg)-9] # remove <result> tag

def clear_chat_history():
    memory_chain = []
    map_chain[userId] = memory_chain
    
def check_grammer(text):
    chat = get_chat()

    if isKorean(text)==True:
        system = (
            "다음의 <article> tag안의 문장의 오류를 찾아서 설명하고, 오류가 수정된 문장을 답변 마지막에 추가하여 주세요."
        )
    else: 
        system = (
            "Here is pieces of article, contained in <article> tags. Find the error in the sentence and explain it, and add the corrected sentence at the end of your answer."
        )
        
    human = "<article>{text}</article>"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    # print('prompt: ', prompt)
    
    chain = prompt | chat    
    try: 
        result = chain.invoke(
            {
                "text": text
            }
        )
        
        msg = result.content
        print('result of grammer correction: ', msg)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)        
        raise Exception ("Not able to request to LLM")
    
    return msg

def upload_to_s3(file_bytes, file_name):
    """
    Upload a file to S3 and return the URL
    """
    try:
        s3_client = boto3.client(
            service_name='s3',
            region_name=bedrock_region
        )

        # Generate a unique file name to avoid collisions
        #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        #unique_id = str(uuid.uuid4())[:8]
        #s3_key = f"uploaded_images/{timestamp}_{unique_id}_{file_name}"
        s3_key = f"{s3_prefix}/{file_name}"

        content_type = (
            "image/jpeg"
            if file_name.lower().endswith((".jpg", ".jpeg"))
            else "image/png"
        )

        s3_client.put_object(
            Bucket=bucketName, Key=s3_key, Body=file_bytes, ContentType=content_type
        )

        url = f"https://{bucketName}.s3.amazonaws.com/{s3_key}"
        return url
    
    except Exception as e:
        err_msg = f"Error uploading to S3: {str(e)}"
        print(err_msg)
        return None

def extract_and_display_s3_images(text, s3_client):
    """
    Extract S3 URLs from text, download images, and return them for display
    """
    s3_pattern = r"https://[\w\-\.]+\.s3\.amazonaws\.com/[\w\-\./]+"
    s3_urls = re.findall(s3_pattern, text)

    images = []
    for url in s3_urls:
        try:
            bucket = url.split(".s3.amazonaws.com/")[0].split("//")[1]
            key = url.split(".s3.amazonaws.com/")[1]

            response = s3_client.get_object(Bucket=bucket, Key=key)
            image_data = response["Body"].read()

            image = Image.open(BytesIO(image_data))
            images.append(image)

        except Exception as e:
            err_msg = f"Error downloading image from S3: {str(e)}"
            print(err_msg)
            continue

    return images

def summary_image(object_name, prompt):
    # load image
    s3_client = boto3.client(
        service_name='s3',
        region_name=bedrock_region
    )
                    
    image_obj = s3_client.get_object(Bucket=bucketName, Key=s3_prefix+'/'+object_name)
    # print('image_obj: ', image_obj)
    
    image_content = image_obj['Body'].read()
    img = Image.open(BytesIO(image_content))
    
    width, height = img.size 
    print(f"width: {width}, height: {height}, size: {width*height}")
    
    isResized = False
    while(width*height > 5242880):                    
        width = int(width/2)
        height = int(height/2)
        isResized = True
        print(f"width: {width}, height: {height}, size: {width*height}")
    
    if isResized:
        img = img.resize((width, height))
    
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    if prompt == "":
        command = "이미지에 포함된 내용을 요약해 주세요."
    else:
        command = prompt
    
    # verify the image
    msg = use_multimodal(img_base64, command)

    return msg, img_base64

def use_multimodal(img_base64, query):    
    multimodal = get_chat()
    
    if query == "":
        query = "그림에 대해 상세히 설명해줘."
    
    messages = [
        SystemMessage(content="답변은 500자 이내의 한국어로 설명해주세요."),
        HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}", 
                    },
                },
                {
                    "type": "text", "text": query
                },
            ]
        )
    ]
    
    try: 
        result = multimodal.invoke(messages)
        
        summary = result.content
        print('result of code summarization: ', summary)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)                    
        raise Exception ("Not able to request to LLM")
    
    return summary

def extract_text(img_base64):    
    multimodal = get_chat()
    query = "그림의 텍스트와 표를 Markdown Format으로 추출합니다."
    
    messages = [
        HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}", 
                    },
                },
                {
                    "type": "text", "text": query
                },
            ]
        )
    ]
    
    try: 
        result = multimodal.invoke(messages)
        
        extracted_text = result.content
        print('result of text extraction from an image: ', extracted_text)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)                    
        raise Exception ("Not able to request to LLM")
    
    #extracted_text = text[text.find('<result>')+8:text.find('</result>')] # remove <result> tag
    print('extracted_text: ', extracted_text)
    if len(extracted_text)>10:
        msg = f"{extracted_text}\n"    
    else:
        msg = "텍스트를 추출하지 못하였습니다."    
    
    return msg

####################### Prompt Flow #######################
# Prompt Flow
###########################################################  
    
flow_arn = None
flow_alias_identifier = None
def run_flow(text, connectionId, requestId):    
    print('prompt_flow_name: ', prompt_flow_name)
    
    client = boto3.client(
        service_name='bedrock-agent',
        region_name=bedrock_region
    )   
    
    global flow_arn, flow_alias_identifier
    
    if not flow_arn:
        response = client.list_flows(
            maxResults=10
        )
        print('response: ', response)
        
        for flow in response["flowSummaries"]:
            print('flow: ', flow)
            if flow["name"] == prompt_flow_name:
                flow_arn = flow["arn"]
                print('flow_arn: ', flow_arn)
                break

    msg = ""
    if flow_arn:
        if not flow_alias_identifier:
            # get flow alias arn
            response_flow_aliases = client.list_flow_aliases(
                flowIdentifier=flow_arn
            )
            print('response_flow_aliases: ', response_flow_aliases)
            
            flowAlias = response_flow_aliases["flowAliasSummaries"]
            for alias in flowAlias:
                print('alias: ', alias)
                if alias['name'] == "latest_version":  # the name of prompt flow alias
                    flow_alias_identifier = alias['arn']
                    print('flowAliasIdentifier: ', flow_alias_identifier)
                    break
        
        # invoke_flow        
        client_runtime = boto3.client(
            service_name='bedrock-agent-runtime',
            region_name=bedrock_region
        )

        response = client_runtime.invoke_flow(
            flowIdentifier=flow_arn,
            flowAliasIdentifier=flow_alias_identifier,
            inputs=[
                {
                    "content": {
                        "document": text,
                    },
                    "nodeName": "FlowInputNode",
                    "nodeOutputName": "document"
                }
            ]
        )
        print('response of invoke_flow(): ', response)
        
        response_stream = response['responseStream']
        try:
            result = {}
            for event in response_stream:
                print('event: ', event)
                result.update(event)
            print('result: ', result)

            if result['flowCompletionEvent']['completionReason'] == 'SUCCESS':
                print("Prompt flow invocation was successful! The output of the prompt flow is as follows:\n")
                # msg = result['flowOutputEvent']['content']['document']
                
                msg = result['flowOutputEvent']['content']['document']
                print('msg: ', msg)
            else:
                print("The prompt flow invocation completed because of the following reason:", result['flowCompletionEvent']['completionReason'])
        except Exception as e:
            raise Exception("unexpected event.",e)

    return msg

rag_flow_arn = None
rag_flow_alias_identifier = None
def run_RAG_prompt_flow(text, connectionId, requestId):
    global rag_flow_arn, rag_flow_alias_identifier
    
    print('rag_prompt_flow_name: ', rag_prompt_flow_name) 
    print('rag_flow_arn: ', rag_flow_arn)
    print('rag_flow_alias_identifier: ', rag_flow_alias_identifier)
    
    client = boto3.client(
        service_name='bedrock-agent',
        region_name=bedrock_region
    )
    if not rag_flow_arn:
        response = client.list_flows(
            maxResults=10
        )
        print('response: ', response)
         
        for flow in response["flowSummaries"]:
            if flow["name"] == rag_prompt_flow_name:
                rag_flow_arn = flow["arn"]
                print('rag_flow_arn: ', rag_flow_arn)
                break
    
    if not rag_flow_alias_identifier and rag_flow_arn:
        # get flow alias arn
        response_flow_aliases = client.list_flow_aliases(
            flowIdentifier=rag_flow_arn
        )
        print('response_flow_aliases: ', response_flow_aliases)
        rag_flow_alias_identifier = ""
        flowAlias = response_flow_aliases["flowAliasSummaries"]
        for alias in flowAlias:
            print('alias: ', alias)
            if alias['name'] == "latest_version":  # the name of prompt flow alias
                rag_flow_alias_identifier = alias['arn']
                print('flowAliasIdentifier: ', rag_flow_alias_identifier)
                break
    
    # invoke_flow
    client_runtime = boto3.client(
        service_name='bedrock-agent-runtime',
        region_name=bedrock_region
    )
    response = client_runtime.invoke_flow(
        flowIdentifier=rag_flow_arn,
        flowAliasIdentifier=rag_flow_alias_identifier,
        inputs=[
            {
                "content": {
                    "document": text,
                },
                "nodeName": "FlowInputNode",
                "nodeOutputName": "document"
            }
        ]
    )
    print('response of invoke_flow(): ', response)
    
    response_stream = response['responseStream']
    try:
        result = {}
        for event in response_stream:
            print('event: ', event)
            result.update(event)
        print('result: ', result)

        if result['flowCompletionEvent']['completionReason'] == 'SUCCESS':
            print("Prompt flow invocation was successful! The output of the prompt flow is as follows:\n")
            # msg = result['flowOutputEvent']['content']['document']
            
            msg = result['flowOutputEvent']['content']['document']
            print('msg: ', msg)
        else:
            print("The prompt flow invocation completed because of the following reason:", result['flowCompletionEvent']['completionReason'])
    except Exception as e:
        raise Exception("unexpected event.",e)

    return msg

####################### Bedrock Agent #######################
# Bedrock Agent
#############################################################

agent_id = agent_alias_id = None
sessionId = dict() 
def run_bedrock_agent(text, connectionId, requestId, userId, sessionState):
    global agent_id, agent_alias_id
    print('agent_id: ', agent_id)
    print('agent_alias_id: ', agent_alias_id)
    
    client = boto3.client(service_name='bedrock-agent')  
    if not agent_id:
        response_agent = client.list_agents(
            maxResults=10
        )
        print('response of list_agents(): ', response_agent)
        
        for summary in response_agent["agentSummaries"]:
            if summary["agentName"] == "tool-executor":
                agent_id = summary["agentId"]
                print('agent_id: ', agent_id)
                break
    
    if not agent_alias_id and agent_id:
        response_agent_alias = client.list_agent_aliases(
            agentId = agent_id,
            maxResults=10
        )
        print('response of list_agent_aliases(): ', response_agent_alias)   
        
        for summary in response_agent_alias["agentAliasSummaries"]:
            if summary["agentAliasName"] == "latest_version":
                agent_alias_id = summary["agentAliasId"]
                print('agent_alias_id: ', agent_alias_id) 
                break
    
    global sessionId
    if not userId in sessionId:
        sessionId[userId] = str(uuid.uuid4())
        
    msg = msg_contents = ""
    if agent_alias_id and agent_id:
        client_runtime = boto3.client('bedrock-agent-runtime')
        try:
            if sessionState:
                response = client_runtime.invoke_agent( 
                    agentAliasId=agent_alias_id,
                    agentId=agent_id,
                    inputText=text, 
                    sessionId=sessionId[userId], 
                    memoryId='memory-'+userId,
                    sessionState=sessionState
                )
            else:
                response = client_runtime.invoke_agent( 
                    agentAliasId=agent_alias_id,
                    agentId=agent_id,
                    inputText=text, 
                    sessionId=sessionId[userId], 
                    memoryId='memory-'+userId
                )
            print('response of invoke_agent(): ', response)
            
            response_stream = response['completion']
            
            for event in response_stream:
                chunk = event.get('chunk')
                if chunk:
                    msg += chunk.get('bytes').decode()
                    print('event: ', chunk.get('bytes').decode())
                        
                    result = {
                        'request_id': requestId,
                        'msg': msg,
                        'status': 'proceeding'
                    }
                    #print('result: ', json.dumps(result))
                    msg = result
                    print('msg: ', msg)
                    
                # files generated by code interpreter
                if 'files' in event:
                    files = event['files']['files']
                    for file in files:
                        objectName = file['name']
                        print('objectName: ', objectName)
                        contentType = file['type']
                        print('contentType: ', contentType)
                        bytes_data = file['bytes']
                                                
                        pixels = BytesIO(bytes_data)
                        pixels.seek(0, 0)
                                    
                        img_key = 'agent/contents/'+objectName
                        
                        s3_client = boto3.client('s3')  
                        response = s3_client.put_object(
                            Bucket=bucketName,
                            Key=img_key,
                            ContentType=contentType,
                            Body=pixels
                        )
                        print('response: ', response)
                        
                        url = path+'agent/contents/'+parse.quote(objectName)
                        print('url: ', url)
                        
                        if contentType == 'application/json':
                            msg_contents = f"\n\n<a href={url} target=_blank>{objectName}</a>"
                        elif contentType == 'application/csv':
                            msg_contents = f"\n\n<a href={url} target=_blank>{objectName}</a>"
                        else:
                            width = 600            
                            msg_contents = f'\n\n<img src=\"{url}\" alt=\"{objectName}\" width=\"{width}\">'
                            print('msg_contents: ', msg_contents)
                                                            
        except Exception as e:
            raise Exception("unexpected event.",e)
        
    return msg+msg_contents
