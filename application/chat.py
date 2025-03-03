import traceback
import boto3
import os
import json
import re
import uuid
import time
import base64
import info 
import PyPDF2
import csv
import knowledge_base as kb
import utils

from io import BytesIO
from PIL import Image
from langchain_aws import ChatBedrock
from botocore.config import Config
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.tools import tool
from langchain.docstore.document import Document
from tavily import TavilyClient  
from langchain_community.tools.tavily_search import TavilySearchResults
from urllib import parse
from pydantic.v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage

logger = utils.CreateLogger("chat")

userId = "demo"
map_chain = dict() 

def initiate():
    global userId
    global memory_chain

    userId = uuid.uuid4().hex
    logger.info(f"userId: {userId}")

    if userId in map_chain:  
            # print('memory exist. reuse it!')
            memory_chain = map_chain[userId]
    else: 
        # print('memory does not exist. create new one!')        
        memory_chain = ConversationBufferWindowMemory(memory_key="chat_history", output_key='answer', return_messages=True, k=5)
        map_chain[userId] = memory_chain

initiate()

try:
    with open("/home/config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
        logger.info(f"config: {config}")

except Exception:
    logger.info(f"use local configuration")
    with open("application/config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
        logger.info(f"config: {config}")

bedrock_region = config["region"] if "region" in config else "us-west-2"

projectName = config["projectName"] if "projectName" in config else "bedrock-agent"

accountId = config["accountId"] if "accountId" in config else None
if accountId is None:
    raise Exception ("No accountId")

region = config["region"] if "region" in config else "us-west-2"
logger.info(f"region: {region}")

s3_prefix = 'docs'
s3_image_prefix = 'images'

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

path = config["sharing_url"] if "sharing_url" in config else None
if path is None:
    raise Exception ("No Sharing URL")

agent_role_arn = config["agent_role_arn"] if "agent_role_arn" in config else None
if agent_role_arn is None:
    raise Exception ("No agent role")

s3_arn = config["s3_arn"] if "s3_arn" in config else None
if s3_arn is None:
    raise Exception ("No S3 ARN")

s3_bucket = config["s3_bucket"] if "s3_bucket" in config else None
if s3_bucket is None:
    raise Exception ("No storage!")

lambda_tools_arn = config["lambda-tools"] if "lambda-tools" in config else None
if lambda_tools_arn is None:
    raise Exception ("No Lambda Tool")

parsingModelArn = f"arn:aws:bedrock:{region}::foundation-model/anthropic.claude-3-haiku-20240307-v1:0"
embeddingModelArn = f"arn:aws:bedrock:{region}::foundation-model/amazon.titan-embed-text-v2:0"

knowledge_base_name = projectName

prompt_flow_name = 'aws-bot'
rag_prompt_flow_name = 'rag-prompt-flow'
knowledge_base_name = projectName

numberOfDocs = 4
MSG_LENGTH = 100    
grade_state = "LLM" # LLM, OTHERS

doc_prefix = s3_prefix+'/'

model_name = "Claude 3.5 Sonnet"
model_type = "claude"
models = info.get_model_info(model_name)
model_id = models[0]["model_id"]
debug_mode = "Enable"

agent_id = agent_alias_id = None
agent_name = projectName
agent_alias_name = "latest_version"
agent_alias_arn = ""

agent_kb_id = agent_kb_alias_id = None
agent_kb_name = projectName+'-knowledge-base'
agent_kb_alias_name = "latest_version"
agent_kb_alias_arn = ""

action_group_name = f"Tools"
action_group_name_for_multi_agent = f"MultiAgentTools"

client = boto3.client(
    service_name='bedrock-agent',
    region_name=bedrock_region
)  

def update(modelName, debugMode, st):    
    global model_name, model_id, model_type, debug_mode
    global models, agent_id, agent_kb_id
    global agent_alias_id, agent_kb_alias_id, agent_alias_arn, agent_kb_alias_arn
    
    if model_name != modelName:
        model_name = modelName
        logger.info(f"model_name: {model_name}")
        
        models = info.get_model_info(model_name)
        model_id = models[0]["model_id"]
        model_type = models[0]["model_type"]

        # retrieve agent_id
        agent_id = retrieve_agent_id(agent_name)
        logger.info(f"agent_id: {agent_id}")
        
        # update agent
        if agent_id: 
            agent_alias_id = update_agent(model_id, model_name, agent_id, agent_name, agent_alias_id, agent_alias_name, st)
        else:
            agent_id, agent_alias_id, agent_alias_arn = create_bedrock_agent(model_id, model_name, "Disable", agent_name, agent_alias_name, st)
        
        # retrieve agent_kb_id
        agent_kb_id = retrieve_agent_id(agent_kb_name)
        logger.info(f"agent_kb_id: {agent_kb_id}")

        # update agent (kb)
        if agent_kb_id: 
            agent_kb_alias_id = update_agent(model_id, model_name, agent_kb_id, agent_kb_name, agent_kb_alias_id, agent_kb_alias_name, st)                        
        else:
            agent_kb_id, agent_kb_alias_id, agent_kb_alias_arn = create_bedrock_agent(model_id, model_name, "Enable", agent_kb_name, agent_kb_alias_name, st)
                                
    if debug_mode != debugMode:
        debug_mode = debugMode
        logger.info(f"debug_mode: {debug_mode}")

def clear_chat_history():
    memory_chain = []
    map_chain[userId] = memory_chain

def save_chat_history(text, msg):
    memory_chain.chat_memory.add_user_message(text)
    if len(msg) > MSG_LENGTH:
        memory_chain.chat_memory.add_ai_message(msg[:MSG_LENGTH])                          
    else:
        memory_chain.chat_memory.add_ai_message(msg) 

def get_chat():
    global model_type

    profile = models[0]
    # print('profile: ', profile)
        
    bedrock_region =  profile['bedrock_region']
    modelId = profile['model_id']    
    model_type = profile['model_type']
    if model_type == 'claude':
        maxOutputTokens = 4096 # 4k
    else: # nova
        maxOutputTokens = 5120 # 5k
    
    logger.info(f"LLM: bedrock_region: {bedrock_region}, modelId: {modelId}, model_type: {model_type}")

    if model_type == 'nova':
        STOP_SEQUENCE = '"\n\n<thinking>", "\n<thinking>", " <thinking>"'
    elif model_type == 'claude':
        STOP_SEQUENCE = "\n\nHuman:" 
                          
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
        "stop_sequences": [STOP_SEQUENCE]
    }
    # print('parameters: ', parameters)

    chat = ChatBedrock(   # new chat model
        model_id=modelId,
        client=boto3_bedrock, 
        model_kwargs=parameters,
        region_name=bedrock_region
    )    
    
    return chat

def print_doc(i, doc):
    if len(doc.page_content)>=100:
        text = doc.page_content[:100]
    else:
        text = doc.page_content
            
    logger.info(f"{i}: {text}, metadata:{doc.metadata}")

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
        logger.info(f"translated text: {msg}")
    except Exception:
        err_msg = traceback.format_exc()
        logger.info(f"error message: {err_msg}")      
        raise Exception ("Not able to request to LLM")

    return msg[msg.find('<result>')+8:len(msg)-9] # remove <result> tag
    
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
        logger.info(f"result of grammer correction: {msg}")
    except Exception:
        err_msg = traceback.format_exc()
        logger.info(f"error message: {err_msg}")       
        raise Exception ("Not able to request to LLM")
    
    return msg

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

def get_retrieval_grader(llm):
    system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )
    
    structured_llm_grader = llm.with_structured_output(GradeDocuments)
    retrieval_grader = grade_prompt | structured_llm_grader
    return retrieval_grader

def grade_documents(question, documents):
    logger.info(f"###### grade_documents ######")
    
    logger.info(f"start grading...")
    logger.info(f"grade_state: {grade_state}")
    
    if grade_state == "LLM":
        filtered_docs = []
        # Score each doc    
        llm = get_chat()
        retrieval_grader = get_retrieval_grader(llm)
        for i, doc in enumerate(documents):
            # print('doc: ', doc)
            print_doc(i, doc)
            
            score = retrieval_grader.invoke({"question": question, "document": doc.page_content})
            # print("score: ", score)
            
            grade = score.binary_score
            # print("grade: ", grade)
            # Document relevant
            if grade.lower() == "yes":
                logger.info(f"---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(doc)
            # Document not relevant
            else:
                logger.info(f"---GRADE: DOCUMENT NOT RELEVANT---")
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
    logger.info(f"length of relevant_docs: {len(docs)}")
    for doc in docs:            
        # print('excerpt: ', doc['metadata']['excerpt'])
            if doc.page_content in contentList:
                logger.info(f"duplicated")
                continue
            contentList.append(doc.page_content)
            updated_docs.append(doc)            
    length_updateed_docs = len(updated_docs)     
    
    if length_original == length_updateed_docs:
        logger.info(f"no duplication")
    
    return updated_docs

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
tavily_key = tavily_api_wrapper = ""
try:
    get_tavily_api_secret = secretsmanager.get_secret_value(
        SecretId=f"tavilyapikey-{projectName}"
    )
    #print('get_tavily_api_secret: ', get_tavily_api_secret)
    secret = json.loads(get_tavily_api_secret['SecretString'])
    #print('secret: ', secret)

    if "tavily_api_key" in secret:
        tavily_key = secret['tavily_api_key']
        #print('tavily_api_key: ', tavily_api_key)

        if tavily_key:
            tavily_api_wrapper = TavilySearchAPIWrapper(tavily_api_key=tavily_key)
            #     os.environ["TAVILY_API_KEY"] = tavily_key

            # # Tavily Tool Test
            # query = 'what is Amazon Nova Pro?'
            # search = TavilySearchResults(
            #     max_results=1,
            #     include_answer=True,
            #     include_raw_content=True,
            #     api_wrapper=tavily_api_wrapper,
            #     search_depth="advanced", # "basic"
            #     # include_domains=["google.com", "naver.com"]
            # )
            # output = search.invoke(query)
            # print('tavily output: ', output)    
        else:
            logger.info(f"tavily_key is required.")
except Exception as e: 
    logger.info(f"Tavily credential is required: {e}")
    raise e

def get_references(docs):    
    reference = ""
    for i, doc in enumerate(docs):
        page = ""
        if "page" in doc.metadata:
            page = doc.metadata['page']
            #print('page: ', page)            
        url = ""
        if "url" in doc.metadata:
            url = doc.metadata['url']
            logger.info(f"url: {url}")
        name = ""
        if "name" in doc.metadata:
            name = doc.metadata['name']
            #print('name: ', name)     
        
        sourceType = ""
        if "from" in doc.metadata:
            sourceType = doc.metadata['from']
        else:
            # if useEnhancedSearch:
            #     sourceType = "OpenSearch"
            # else:
            #     sourceType = "WWW"
            sourceType = "WWW"

        #print('sourceType: ', sourceType)        
        
        #if len(doc.page_content)>=1000:
        #    excerpt = ""+doc.page_content[:1000]
        #else:
        #    excerpt = ""+doc.page_content
        excerpt = ""+doc.page_content
        # print('excerpt: ', excerpt)
        
        # for some of unusual case 
        #excerpt = excerpt.replace('"', '')        
        #excerpt = ''.join(c for c in excerpt if c not in '"')
        excerpt = re.sub('"', '', excerpt)
        excerpt = re.sub('#', '', excerpt)     
        excerpt = re.sub('\n', '', excerpt)      
        logger.info(f"excerpt(quotation removed): {excerpt}")
        
        if page:                
            reference += f"{i+1}. {page}page in [{name}]({url})), {excerpt[:30]}...\n"
        else:
            reference += f"{i+1}. [{name}]({url}), {excerpt[:30]}...\n"

    if reference: 
        reference = "\n\n#### 관련 문서\n"+reference

    return reference

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
        logger.info(f"Exception: {e}")

    return docs

def isKorean(text):
    # check korean
    pattern_hangul = re.compile('[\u3131-\u3163\uac00-\ud7a3]+')
    word_kor = pattern_hangul.search(str(text))
    # print('word_kor: ', word_kor)

    if word_kor and word_kor != 'None':
        # logger.info(f"Korean: {word_kor}")
        return True
    else:
        # logger.info(f"Not Korean:: {word_kor}")
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
        logger.info(f"error message: {err_msg}")     
        raise Exception ("Not able to request to LLM")

    return msg[msg.find('<result>')+8:len(msg)-9] # remove <result> tag


####################### LangChain #######################
# General Conversation
#########################################################
def general_conversation(query):
    llm = get_chat()

    system = (
        "당신의 이름은 서연이고, 질문에 대해 친절하게 답변하는 사려깊은 인공지능 도우미입니다."
        "상황에 맞는 구체적인 세부 정보를 충분히 제공합니다." 
        "모르는 질문을 받으면 솔직히 모른다고 말합니다."
    )
    
    human = "Question: {input}"
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system), 
        MessagesPlaceholder(variable_name="history"), 
        ("human", human)
    ])
                
    history = memory_chain.load_memory_variables({})["chat_history"]

    chain = prompt | llm | StrOutputParser()
    try: 
        stream = chain.stream(
            {
                "history": history,
                "input": query,
            }
        )  
        logger.info(f"stream: {stream}")
            
    except Exception:
        err_msg = traceback.format_exc()
        logger.info(f"error message: {err_msg}")      
        raise Exception ("Not able to request to LLM: "+err_msg)
        
    return stream

def get_rag_prompt(text):
    # print("###### get_rag_prompt ######")
    llm = get_chat()
    # print('model_type: ', model_type)
    
    if model_type == "nova":
        if isKorean(text)==True:
            system = (
                "당신의 이름은 서연이고, 질문에 대해 친절하게 답변하는 사려깊은 인공지능 도우미입니다."
                "다음의 Reference texts을 이용하여 user의 질문에 답변합니다."
                "모르는 질문을 받으면 솔직히 모른다고 말합니다."
                "답변의 이유를 풀어서 명확하게 설명합니다."
            )
        else: 
            system = (
                "You will be acting as a thoughtful advisor."
                "Provide a concise answer to the question at the end using reference texts." 
                "If you don't know the answer, just say that you don't know, don't try to make up an answer."
                "You will only answer in text format, using markdown format is not allowed."
            )    
    
        human = (
            "Question: {question}"

            "Reference texts: "
            "{context}"
        ) 
        
    elif model_type == "claude":
        if isKorean(text)==True:
            system = (
                "당신의 이름은 서연이고, 질문에 대해 친절하게 답변하는 사려깊은 인공지능 도우미입니다."
                "다음의 <context> tag안의 참고자료를 이용하여 상황에 맞는 구체적인 세부 정보를 충분히 제공합니다." 
                "모르는 질문을 받으면 솔직히 모른다고 말합니다."
                "답변의 이유를 풀어서 명확하게 설명합니다."
                "결과는 <result> tag를 붙여주세요."
            )
        else: 
            system = (
                "You will be acting as a thoughtful advisor."
                "Here is pieces of context, contained in <context> tags." 
                "If you don't know the answer, just say that you don't know, don't try to make up an answer."
                "You will only answer in text format, using markdown format is not allowed."
                "Put it in <result> tags."
            )    

        human = (
            "<question>"
            "{question}"
            "</question>"

            "<context>"
            "{context}"
            "</context>"
        )

    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    # print('prompt: ', prompt)
    
    rag_chain = prompt | llm

    return rag_chain

def run_rag_with_knowledge_base(text, st):
    global reference_docs, contentList
    reference_docs = []
    contentList = []

    msg = ""
    top_k = numberOfDocs
    
    # retrieve
    if debug_mode == "Enable":
        st.info(f"RAG 검색을 수행합니다. 검색어: {text}")  
    
    relevant_docs = kb.retrieve_documents_from_knowledge_base(text, top_k=top_k)
    # relevant_docs += retrieve_documents_from_tavily(text, top_k=top_k)

    # grade   
    if debug_mode == "Enable":
        st.info(f"가져온 {len(relevant_docs)}개의 문서를 평가하고 있습니다.") 

    # docs = []
    # for doc in relevant_docs:
    #     chat = get_chat()
    #     if not isKorean(doc.page_content):
    #         translated_content = traslation(chat, doc.page_content, "English", "Korean")
    #         doc.page_content = translated_content
    #         print("doc.page_content: ", doc.page_content)
    #     docs.append(doc)
    # print('translated relevant docs: ', docs)

    filtered_docs = grade_documents(text, relevant_docs)
    
    filtered_docs = check_duplication(filtered_docs) # duplication checker

    if len(filtered_docs):
        reference_docs += filtered_docs 

    if debug_mode == "Enable":
        st.info(f"{len(filtered_docs)}개의 문서가 선택되었습니다.")
    
    # generate
    if debug_mode == "Enable":
        st.info(f"결과를 생성중입니다.")
    relevant_context = ""
    for document in filtered_docs:
        relevant_context = relevant_context + document.page_content + "\n\n"        
    # print('relevant_context: ', relevant_context)

    rag_chain = get_rag_prompt(text)
                       
    msg = ""    
    try: 
        result = rag_chain.invoke(
            {
                "question": text,
                "context": relevant_context                
            }
        )
        logger.info(f"result: {result}")

        msg = result.content        
        if msg.find('<result>')!=-1:
            msg = msg[msg.find('<result>')+8:msg.find('</result>')]
        
    except Exception:
        err_msg = traceback.format_exc()
        logger.info(f"error message: {err_msg}")                    
        raise Exception ("Not able to request to LLM")
    
    reference = ""
    if reference_docs:
        reference = get_references(reference_docs)

    return msg+reference, reference_docs


####################### Prompt Flow #######################
# Prompt Flow
###########################################################  
flow_arn = None
flow_alias_identifier = None
def run_flow(text, connectionId, requestId):    
    logger.info(f"prompt_flow_name: {prompt_flow_name}")
    
    global flow_arn, flow_alias_identifier
    
    if not flow_arn:
        response = client.list_flows(
            maxResults=10
        )
        logger.info(f"response: {response}")
        
        for flow in response["flowSummaries"]:
            logger.info(f"flow: {flow}")
            if flow["name"] == prompt_flow_name:
                flow_arn = flow["arn"]
                logger.info(f"flow_arn: {flow_arn}")
                break

    msg = ""
    if flow_arn:
        if not flow_alias_identifier:
            # get flow alias arn
            response_flow_aliases = client.list_flow_aliases(
                flowIdentifier=flow_arn
            )
            logger.info(f"response_flow_aliases: {response_flow_aliases}")
            
            flowAlias = response_flow_aliases["flowAliasSummaries"]
            for alias in flowAlias:
                logger.info(f"alias: {alias}")
                if alias['name'] == "latest_version":  # the name of prompt flow alias
                    flow_alias_identifier = alias['arn']
                    logger.info(f"flowAliasIdentifier: {flow_alias_identifier}")
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
        logger.info(f"response of invoke_flow(): {response}")
        
        response_stream = response['responseStream']
        try:
            result = {}
            for event in response_stream:
                logger.info(f"event: {event}")
                result.update(event)
            logger.info(f"result: {result}")

            if result['flowCompletionEvent']['completionReason'] == 'SUCCESS':
                logger.info(f"Prompt flow invocation was successful! The output of the prompt flow is as follows:\n")
                # msg = result['flowOutputEvent']['content']['document']
                
                msg = result['flowOutputEvent']['content']['document']
                logger.info(f"msg: {msg}")
            else:
                logger.info(f"The prompt flow invocation completed because of the following reason:", result['flowCompletionEvent']['completionReason'])
        except Exception as e:
            raise Exception("unexpected event.",e)

    return msg

rag_flow_arn = None
rag_flow_alias_identifier = None
def run_RAG_prompt_flow(text, connectionId, requestId):
    global rag_flow_arn, rag_flow_alias_identifier
    
    logger.info(f"rag_prompt_flow_name: {rag_prompt_flow_name}")
    logger.info(f"rag_flow_arn: {rag_flow_arn}")
    logger.info(f"rag_flow_alias_identifier: {rag_flow_alias_identifier}")
    
    if not rag_flow_arn:
        response = client.list_flows(
            maxResults=10
        )
        logger.info(f"response: {response}")
         
        for flow in response["flowSummaries"]:
            if flow["name"] == rag_prompt_flow_name:
                rag_flow_arn = flow["arn"]
                logger.info(f"rag_flow_arn: {rag_flow_arn}")
                break
    
    if not rag_flow_alias_identifier and rag_flow_arn:
        # get flow alias arn
        response_flow_aliases = client.list_flow_aliases(
            flowIdentifier=rag_flow_arn
        )
        logger.info(f"response_flow_aliases: {response_flow_aliases}")
        rag_flow_alias_identifier = ""
        flowAlias = response_flow_aliases["flowAliasSummaries"]
        for alias in flowAlias:
            logger.info(f"alias: {alias}")
            if alias['name'] == "latest_version":  # the name of prompt flow alias
                rag_flow_alias_identifier = alias['arn']
                logger.info(f"flowAliasIdentifier: {rag_flow_alias_identifier}")
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
    logger.info(f"response of invoke_flow(): {response}")
    
    response_stream = response['responseStream']
    try:
        result = {}
        for event in response_stream:
            logger.info(f"event: {event}")
            result.update(event)
        logger.info(f"result: {result}")

        if result['flowCompletionEvent']['completionReason'] == 'SUCCESS':
            logger.info(f"Prompt flow invocation was successful! The output of the prompt flow is as follows:\n")
            # msg = result['flowOutputEvent']['content']['document']
            
            msg = result['flowOutputEvent']['content']['document']
            logger.info(f"msg: {msg}")
        else:
            logger.info(f"The prompt flow invocation completed because of the following reason:", result['flowCompletionEvent']['completionReason'])
    except Exception as e:
        raise Exception("unexpected event.",e)

    return msg

####################### Bedrock Agent #######################
# Bedrock Agent (Single)
########################################################### 
sessionId = dict() 
sessionState = ""

def show_output(event, st):
    global reference_docs
    stream_result = final_result = ""    
    image_url = []
        
    #logger.info(str(event))
    #print("\n")
    #logger.info(f"\n")
        
    # Handle text chunks
    if "chunk" in event:
        chunk = event["chunk"]
        if "bytes" in chunk:
            text = chunk["bytes"].decode("utf-8")
            logger.info(f"Chunk: {text}")
            stream_result += text

    # Handle file outputs
    if "files" in event:
        logger.info(f"Files received")
        files = event["files"]["files"]

        logger.info(f"Number of files: {len(files)}")
        for i, file in enumerate(files):
            output_type = file["type"]

            if output_type == "image/jpeg" or output_type == "image/png":
                st.image(file["bytes"], caption=file["name"])
                logger.info(f"image[{i}]: {file['name']}")

                file_url = upload_to_s3(file["bytes"], file["name"])
                logger.info(f"file_url[{i}]: {file_url}")

                file_name = file_url[file_url.rfind('/')+1:]
                url = f"{path}/{s3_image_prefix}/{file_name}"
                logger.info(f"(files) image_url[{i}]: {url}")
                image_url.append(url)
            elif output_type=="text/csv":
                st.info(file["bytes"].decode("utf-8"))
                logger.info(f"csv[{i}]: {file['name']}")
            else:
                logger.info(f"unexpected type: {file['name']}, {file['type']}")

    # Check trace
    if "trace" in event:
        if ("trace" in event["trace"] and "orchestrationTrace" in event["trace"]["trace"]):
            trace_event = event["trace"]["trace"]["orchestrationTrace"]
            if "rationale" in trace_event:
                trace_text = trace_event["rationale"]["text"]
                if debug_mode=="Enable":
                    st.info(f"rationale: {trace_text}")

            if "modelInvocationInput" in trace_event:
                if "text" in trace_event["modelInvocationInput"]:
                    trace_text = trace_event["modelInvocationInput"]["text"]
                    logger.info(f"trace_text: {trace_text}")
                    #if debug_mode=="Enable":
                        # st.info(f"modelInvocationInput: {trace_text}")
                if "rawResponse" in trace_event["modelInvocationInput"]:
                    rawResponse = trace_event["modelInvocationInput"]["rawResponse"]                        
                    logger.info(f"rawResponse: {rawResponse}")
                    # if debug_mode=="Enable":
                    #     st.info(f"modelInvocationInput: {rawResponse}")

            if "modelInvocationOutput" in trace_event:
                if "rawResponse" in trace_event["modelInvocationOutput"]:
                    trace_text = trace_event["modelInvocationOutput"]["rawResponse"]["content"]
                    logger.info(f"trace_text: {trace_text}")
                    # if debug_mode=="Enable":
                    #     st.info(f"modelInvocationOutput: {trace_text}")

            if "invocationInput" in trace_event:
                if "codeInterpreterInvocationInput" in trace_event["invocationInput"]:
                    trace_code = trace_event["invocationInput"]["codeInterpreterInvocationInput"]["code"]
                    logger.info(f"trace_code: {trace_code}")
                    if debug_mode=="Enable":
                        st.info(f"codeInterpreter: {trace_code}")

                if "knowledgeBaseLookupInput" in trace_event["invocationInput"]:
                    trace_text = trace_event["invocationInput"]["knowledgeBaseLookupInput"]["text"]
                    logger.info(f"trace_text: {trace_text}")
                    # st.info(f"knowledgeBaseLookup: {trace_text}")
                    if debug_mode=="Enable":
                        st.info(f"RAG를 검색합니다. 검색어: {trace_text}")

                if "actionGroupInvocationInput" in trace_event["invocationInput"]:
                    trace_function = trace_event["invocationInput"]["actionGroupInvocationInput"]["function"]
                    logger.info(f"preptrace_functionare: {trace_function}")
                    if debug_mode=="Enable":
                        st.info(f"actionGroupInvocation: {trace_function}")

            if "observation" in trace_event:
                if "finalResponse" in trace_event["observation"]:
                    trace_resp = trace_event["observation"]["finalResponse"]["text"]
                    logger.info(f"final response: {trace_resp}")   
                    final_result = trace_resp

                if ("codeInterpreterInvocationOutput" in trace_event["observation"]):
                    if "executionOutput" in trace_event["observation"]["codeInterpreterInvocationOutput"]:
                        trace_resp = trace_event["observation"]["codeInterpreterInvocationOutput"]["executionOutput"]
                        if debug_mode=="Enable":
                            st.info(f"observation: {trace_resp}")

                    if "executionError" in trace_event["observation"]["codeInterpreterInvocationOutput"]:
                        trace_resp = trace_event["observation"]["codeInterpreterInvocationOutput"]["executionError"]
                        if debug_mode=="Enable":
                            st.error(f"observation: {trace_resp}")

                        if "image_url" in trace_resp:
                            file_url = trace_resp["image_url"]
                            logger.info(f"file_url: {file_url}")

                            file_name = file_url[file_url.rfind('/')+1:]
                            url = f"{path}/{s3_image_prefix}/{file_name}"
                            logger.info(f"(observation) image_url: {url}")
                            image_url.append(url)

                            st.image(url)
                            
                if "knowledgeBaseLookupOutput" in trace_event["observation"]:
                    # if debug_mode=="Enable":
                    #     st.info(f"knowledgeBaseLookupOutput: {trace_event["observation"]["knowledgeBaseLookupOutput"]["retrievedReferences"]}")
                    if "retrievedReferences" in trace_event["observation"]["knowledgeBaseLookupOutput"]:
                        references = trace_event["observation"]["knowledgeBaseLookupOutput"]["retrievedReferences"]
                        if debug_mode=="Enable":
                            st.info(f"{len(references)}개의 문서가 검색되었습니다.")

                        logger.info(f"references: {references}")
                        for i, reference in enumerate(references):
                            content = reference['content']['text']
                            # print('content: ', content)
                            uri = reference['location']['s3Location']['uri']
                            # print('uri: ', uri)

                            name = uri.split('/')[-1]
                            encoded_name = parse.quote(name)
                            url = f"{path}/{doc_prefix}{encoded_name}"
                            # print('url: ', url)

                            logger.info(f"--> {i}: {content[:50]}, {name}, {url}")

                            reference_docs.append(
                                Document(
                                    page_content=content,
                                    metadata={
                                        'name': name,
                                        'url': url,
                                        'from': 'RAG'
                                    },
                                )
                            )    

                if "actionGroupInvocationOutput" in trace_event["observation"]:
                    trace_resp = trace_event["observation"]["actionGroupInvocationOutput"]["text"]
                    if debug_mode=="Enable":
                        st.info(f"actionGroupInvocationOutput: {trace_resp}")

                    logger.info(f"hecking trace resp")
                    print(trace_resp)
                    logger.info(trace_resp)

                    # try to covnert to json
                    try:
                        trace_resp = trace_resp.replace("'", '"')
                        trace_resp = json.loads(trace_resp)
                        logger.info(f"converted to json")
                        logger.info(f"{trace_resp}")

                        # check if image_url is in trace_response, if it is download the image and add it to the images object of mdoel response
                        if "image_url" in trace_resp:
                            logger.info(f"got image")
                            image_url = trace_resp["image_url"]
                            st.image(image_url)
                            
                    except:
                        logger.info(f"not json")
                        pass

        elif "guardrailTrace" in event["trace"]["trace"]:
            guardrail_trace = event["trace"]["trace"]["guardrailTrace"]
            if "inputAssessments" in guardrail_trace:
                assessments = guardrail_trace["inputAssessments"]
                for assessment in assessments:
                    if "contentPolicy" in assessment:
                        filters = assessment["contentPolicy"]["filters"]
                        for filter in filters:
                            if filter["action"] == "BLOCKED":
                                if debug_mode=="Enable":
                                    st.error(f"Guardrail blocked {filter['type']} confidence: {filter['confidence']}")
                    if "topicPolicy" in assessment:
                        topics = assessment["topicPolicy"]["topics"]
                        for topic in topics:
                            if topic["action"] == "BLOCKED":
                                if debug_mode=="Enable":
                                    st.error(f"Guardrail blocked topic {topic['name']}")            

    logger.info(f'image_url: {image_url}')

    if final_result:                
        return final_result, image_url 
    else:
        return stream_result, image_url

def deploy_agent(agentId, agentAliasName):
    agentAliasId = agentAliasArn = ""
    try:
        # retrieve agent alias
        response_agent_alias = client.list_agent_aliases(
            agentId = agentId,
            maxResults=10
        )
        logger.info(f"response of list_agent_aliases(): {response_agent_alias}") 

        for summary in response_agent_alias["agentAliasSummaries"]:
            if summary["agentAliasName"] == agentAliasName:
                agentAliasId = summary["agentAliasId"]                
                logger.info(f"agentAliasId: {agentAliasId}")
                break

        if agentAliasId:
            response = client.delete_agent_alias(
                agentAliasId=agentAliasId,
                agentId=agentId
            )            
            logger.info(f"response of agentAliasId(): {response}")  
        
        # create agent alias 
        response = client.create_agent_alias(
            agentAliasName=agentAliasName,
            agentId=agentId,
            description='the lastest deployment'
        )
        logger.info(f"response of create_agent_alias(): {response}")

        agentAliasId = response['agentAlias']['agentAliasId']
        agentAliasArn = response['agentAlias']['agentAliasArn']
        logger.info(f"agentAliasId: {agentAliasId}, agentAliasArn: {agentAliasArn}")
        
    except Exception:
        err_msg = traceback.format_exc()
        logger.info(f"error message: {err_msg}")   
    
    return agentAliasId, agentAliasArn

def update_agent(modelId, modelName, agentId, agentName, agentAliasId, agentAliasName, st):
    if debug_mode=="Enable":
        st.info(f"{agentName}의 모델을 {modelName}로 업데이트합니다.")

    # update agent
    try:
        agent_instruction = (
            "당신의 이름은 서연이고, 질문에 친근한 방식으로 대답하도록 설계된 대화형 AI입니다. "
            "상황에 맞는 구체적인 세부 정보를 충분히 제공합니다. "
            "모르는 질문을 받으면 솔직히 모른다고 말합니다. "
            "한국어로 답변합니다."
        )
        logger.info(f"modelId: {modelId}")

        response = client.update_agent(
            agentId=agentId,
            agentName=agentName,
            agentResourceRoleArn=agent_role_arn,
            instruction=agent_instruction,
            foundationModel=modelId,
            description=f"Agent의 이름은 {agentName} 입니다. 사용 모델은 {modelName}입니다.",
            idleSessionTTLInSeconds=600
        )
        logger.info(f"response of update_agent(): {response}")
    except Exception:
        err_msg = traceback.format_exc()
        logger.info(f"error message: {err_msg}")

    time.sleep(5)            
    
    # preparing
    if debug_mode=="Enable":
        st.info(f'{agentName}를 사용할 수 있도록 "Prepare"로 설정합니다.')    
    prepare_agent(agentId)    

    # deploy
    if debug_mode=="Enable":
        st.info(f'{agentName}을 {agentAliasName}로 배포합니다.')    
    agentAliasId, agentAliasArn = deploy_agent(agentId, agentAliasName)
    time.sleep(3) 

    global agent_id, agent_alias_id, agent_kb_id, agent_kb_alias_id
    agent_id = agent_alias_id = agent_kb_id = agent_kb_alias_id = ""

    return agentAliasId, agentAliasArn

def create_action_group(agentId, actionGroupName, lambdaToolsArn, functionSchema, st):    
    if debug_mode=="Enable":
        st.info(f"Action Group에 {actionGroupName}이 존재하는지 확인합니다.")

    response = client.list_agent_action_groups(
        agentId=agentId,
        agentVersion='DRAFT',
        maxResults=10
    ) 
    logger.info(f"response of list_agent_action_groups(): {response}")

    actionGroupSummaries = response['actionGroupSummaries']

    isExist = False
    for actionGroup in actionGroupSummaries:
        logger.info(f"actionGroupName: {actionGroup['actionGroupId']}")

        if actionGroup['actionGroupId'] == actionGroupName:
            logger.info(f"action group already exists")
            isExist = True
            break
    
    logger.info(f"isExist: {isExist}")
    if not isExist:
        if debug_mode=="Enable":
            st.info(f"{actionGroupName} Action Group을 생성합니다.")

        response = client.create_agent_action_group(
            actionGroupName=actionGroupName,
            actionGroupState='ENABLED',
            agentId=agentId,
            agentVersion='DRAFT',
            description=f"Action Group의 이름은 {actionGroupName} 입니다.",
            actionGroupExecutor={'lambda': lambdaToolsArn},  
            functionSchema=functionSchema
        )
        logger.info(f"response of create_action_group(): {response}")

def create_action_group_for_code_interpreter(agentId, st):
    actionGroupName = "CodeInterpreter"
    if debug_mode=="Enable":
        st.info(f"Action Group에 {actionGroupName}이 존재하는지 확인합니다.")

    response = client.list_agent_action_groups(
        agentId=agentId,
        agentVersion='DRAFT',
        maxResults=10
    )
    logger.info(f"response of list_agent_action_groups(): {response}")

    actionGroupSummaries = response['actionGroupSummaries']

    isExist = False
    for actionGroup in actionGroupSummaries:
        logger.info(f"actionGroupName: {actionGroup['actionGroupId']}")

        if actionGroup['actionGroupId'] == actionGroupName:
            logger.info(f"action group already exists")
            isExist = True
            break
    
    logger.info(f"isExist: {isExist}")
    if not isExist:
        if debug_mode=="Enable":
            st.info(f"{actionGroupName} Action Group을 생성합니다.")

        response = client.create_agent_action_group(
            actionGroupName=actionGroupName,
            actionGroupState='ENABLED',
            agentId=agentId,
            agentVersion='DRAFT',
            parentActionGroupSignature='AMAZON.CodeInterpreter'
        )
        logger.info(f"response of create_action_group_for_code_interpreter(): {response}")

def prepare_agent(agentId):
    try:
        response = client.prepare_agent(
            agentId=agentId
        )
        logger.info(f"response of prepare_agent(): {response}")  
        time.sleep(5) # delay 5 seconds

    except Exception:
        err_msg = traceback.format_exc()
        logger.info(f"'error message: {err_msg}")   

def create_bedrock_agent(modelId, modelName, enable_knowledge_base, agentName, agentAliasName, st):
    if debug_mode=="Enable":
        st.info(f"Agent를 생성합니다. 사용 모델은 {modelName}입니다.")

    # create agent
    agent_instruction = (
        "당신의 이름은 서연이고, 질문에 친근한 방식으로 대답하도록 설계된 대화형 AI입니다. "
        "상황에 맞는 구체적인 세부 정보를 충분히 제공합니다. "
        "모르는 질문을 받으면 솔직히 모른다고 말합니다. "
    )
    logger.info(f"modelId: {modelId}")

    response = client.create_agent(
        agentResourceRoleArn=agent_role_arn,
        instruction=agent_instruction,
        foundationModel=modelId,
        description=f"Bedrock Agent (Knowledge Base) 입니다. 사용 모델은 {modelName}입니다.",
        agentName=agentName,
        idleSessionTTLInSeconds=600
    )
    logger.info(f"response of create_bedrock_gent(): {response}")

    agentId = response['agent']['agentId']
    logger.info(f"agentId: {agentId}")
    time.sleep(5)   

    # create action group
    functionSchema = {
        'functions': [
            {
                'name': 'book_list',
                'description': 'Search book list by keyword and then return book list',                        
                'parameters': {
                    'keyword': {
                        'description': 'Search keyword',
                        'required': True,
                        'type': 'string'
                    }
                },
                'requireConfirmation': 'DISABLED'
            },
            {
                'name': 'current_time',
                'description': "Returns the current date and time in the specified format such as %Y-%m-%d %H:%M:%S",
                'parameters': {
                    'format': {
                        'description': 'time format of the current time',
                        'required': True,
                        'type': 'string'
                    }
                },
                'requireConfirmation': 'DISABLED'
            },
            {
                'name': 'weather',
                'description': "Retrieve weather information by city name and then return weather statement.",
                'parameters': {
                    'city': {
                        'description': 'the name of city to retrieve',
                        'required': True,
                        'type': 'string'
                    }
                },
                'requireConfirmation': 'DISABLED'
            },
            {
                'name': 'search_internet',
                'description': "Search general information by keyword and then return the result as a string.",
                'parameters': {
                    'keyword': {
                        'description': 'search keyword',
                        'required': True,
                        'type': 'string'
                    }
                },
                'requireConfirmation': 'DISABLED'
            },
            {
                'name': 'search_rag',
                'description': "Search technical information by keyword and then return the result as a string.",
                'parameters': {
                    'keyword': {
                        'description': 'search keyword',
                        'required': True,
                        'type': 'string'
                    }
                },
                'requireConfirmation': 'DISABLED'
            },
            {
                'name': 'stock',
                'description': "Retrieve accurate stock data for a given ticker.",
                'parameters': {
                    'ticker': {
                        'description': 'the ticker to retrieve price history for. In South Korea, a ticker is a 6-digit number.',
                        'required': True,
                        'type': 'string'
                    },
                    'country': {
                        'description': 'the English country name of the stock',
                        'required': True,
                        'type': 'string'
                    }
                },
                'requireConfirmation': 'DISABLED'
            }
        ]
    }
    create_action_group(agentId, action_group_name, lambda_tools_arn, functionSchema, st)     

    # create action group for code_interpreter
    create_action_group_for_code_interpreter(agentId, st)
    
    # associate knowledge base            
    if kb.knowledge_base_id and enable_knowledge_base == "Enable":
        if debug_mode=="Enable":
            st.info("Agent에서 Knowledge Base를 사용할 수 있도록 설정합니다.")

        rag_prompt = (
            "당신의 이름은 서연이고, 질문에 대해 친절하게 답변하는 사려깊은 인공지능 도우미입니다."
            "다음의 Reference texts을 이용하여 user의 질문에 답변합니다."
            "모르는 질문을 받으면 솔직히 모른다고 말합니다."
            "답변의 이유를 풀어서 명확하게 설명합니다."
        )
        try: 
            response = client.associate_agent_knowledge_base(
                agentId=agentId,
                agentVersion='DRAFT',
                description=rag_prompt,
                knowledgeBaseId=kb.knowledge_base_id,
                knowledgeBaseState='ENABLED'
            )
            logger.info(f"response of associate_agent_knowledge_base(): {response}")
            time.sleep(5) # delay 5 seconds
        except Exception:
            err_msg = traceback.format_exc()
            logger.info(f"error message: {err_msg}")

    # preparing
    if debug_mode=="Enable":
        st.info('Agent를 사용할 수 있도록 "Prepare"로 설정합니다.')    
    prepare_agent(agentId)
    
    # deploy
    if debug_mode=="Enable":
        st.info(f'{agentName}을 {agentAliasName}로 배포합니다.')    
    agentAliasId, agentAliasArn = deploy_agent(agentId, agentAliasName)
    time.sleep(3) 

    return agentId, agentAliasId, agentAliasArn           

def retrieve_agent_id(agentName):
    response_agent = client.list_agents(
        maxResults=10
    )
    logger.info(f"response of list_agents(): {response_agent}")

    agentId = ""
    for summary in response_agent["agentSummaries"]:
        if summary["agentName"] == agentName:
            agentId = summary["agentId"]
            logger.info(f"agentId: {agentId}")
            break

    return agentId  

def check_bedrock_agent_status(agentName, agentAliasId, agentAliasName, agentAliasArn, st):
    agentId = retrieve_agent_id(agentName)  
    
    # create agent if no agent
    if not agentId:        
        agentId, agentAliasId, agentAliasArn = create_bedrock_agent(model_id, model_name, "Disable", agentName, agentAliasName, st)           
    # else:
    #     response = client.get_agent(
    #         agentId=agentId
    #     )
    #     print('response of get_agent(): ', response)

    if not agentAliasId and agentId:
        if debug_mode=="Enable":
            st.info('Agent의 alias를 검색합니다.')

        # retrieve agent alias
        response_agent_alias = client.list_agent_aliases(
            agentId = agentId,
            maxResults=10
        )
        logger.info(f"response of list_agent_aliases(): {response_agent_alias}")

        for summary in response_agent_alias["agentAliasSummaries"]:
            if summary["agentAliasName"] == agentAliasName:
                agentAliasId = summary["agentAliasId"]
                logger.info(f"agentAliasId: {agentAliasId}")

                if not agentAliasArn:
                    response = client.get_agent_alias(
                        agentAliasId=agentAliasId,
                        agentId=agentId
                    )
                    logger.info(f"response of get_agent_alias(): {response}")

                    agentAliasArn = response["agentAlias"]["agentAliasArn"]
                    logger.info(f"agentAliasArn: {agentAliasArn}")

                logger.info(f"agentAliasStatus: {summary['agentAliasStatus']}")
                if not summary["agentAliasStatus"] == "PREPARED":
                    if debug_mode=="Enable":
                        st.info('Agent를 사용할 수 있도록 "Prepare"로 다시 설정합니다.')
                    prepare_agent(agentId)
                break
        
        # creat agent alias if no alias
        if not agentAliasId:
            if debug_mode=="Enable":
                st.info('Agent의 alias를 생성합니다.')

            response = client.create_agent_alias(
                agentAliasName=agentAliasName,
                agentId=agentId,
                description='the lastest deployment'
            )
            logger.info(f"response of create_agent_alias(): {response}")

            agentAliasId = response['agentAlias']['agentAliasId']
            logger.info(f"agentAliasId: {agentAliasId}")
            time.sleep(5) # delay 5 seconds

            agentAliasId, agentAliasArn = deploy_agent(agentId, agentAliasName)
            time.sleep(5) # delay 5 seconds
            logger.info(f"agentAliasId: {agentAliasId}, agentAliasArn: {agentAliasArn}")
            
    return agentId, agentAliasId, agentAliasArn

def run_bedrock_agent(text, agentName, sessionState, st):   
    global  agent_id, agent_alias_id, agent_kb_id, agent_kb_alias_id    
    if agentName == agent_name:
        agentId = agent_id
        agentAliasId = agent_alias_id
        agentAliasName = agent_alias_name
        agentAliasArn = agent_alias_arn
    else:
        agentId = agent_kb_id
        agentAliasId = agent_kb_alias_id
        agentAliasName = agent_kb_alias_name
        agentAliasArn = agent_kb_alias_arn

    logger.info(f"agentId: {agentId} agentAliasId: {agentAliasId}")

    if not agentId or not agentAliasId:        
        agentId, agentAliasId, agentAliasArn = check_bedrock_agent_status(agentName, agentAliasId, agentAliasName, agentAliasArn, st)
        logger.info(f"agentId: {agentId} agentAliasId: {agentAliasId}, agentAliasArn: {agentAliasArn}")

        if agentName == agent_name:
            agent_id = agentId
            agent_alias_id = agentAliasId
        else:
            agent_kb_id = agentId
            agent_kb_alias_id = agentAliasId

    global reference_docs
    reference_docs = []

    global sessionId
    if not userId in sessionId:
        sessionId[userId] = str(uuid.uuid4())

    result = ""
    final_result = "" 
    image_url = []
    if agentAliasId and agentId:
        #if debug_mode=="Enable":
        #    st.info('답변을 생성하고 있습니다.')

        client_runtime = boto3.client(
            service_name='bedrock-agent-runtime',
            region_name=bedrock_region
        )
        try:
            if sessionState:
                logger.info(f'code interpreter is used with sessionState')
                response = client_runtime.invoke_agent( 
                    agentAliasId=agentAliasId,
                    agentId=agentId,
                    inputText=text, 
                    enableTrace=True,
                    sessionId=sessionId[userId], 
                    memoryId='memory-'+userId,
                    sessionState=sessionState
                )
            else:
                response = client_runtime.invoke_agent( 
                    agentAliasId=agentAliasId,
                    agentId=agentId,
                    inputText=text, 
                    enableTrace=True,
                    sessionId=sessionId[userId], 
                    memoryId='memory-'+userId
                )
            logger.info(f"response of invoke_agent(): {response}")
            
            response_stream = response['completion']
               
            image_url = []
            for index, event in enumerate(response_stream):
                result, image_url = show_output(event, st)
                if result:
                    logger.info(f"event: {index}, result: {result}")
                    final_result = result
                    
        except Exception as e:
            agent_id = agent_alias_id = agent_kb_id = agent_kb_alias_id = ""
            # raise Exception("unexpected event.",e)
            if debug_mode=="Enable":
                st.error('실패하여 agent 정보를 초기화하였습니다. 재시도해주세요.')
            err_msg = traceback.format_exc()
            logger.info(f"error message: {err_msg}")
                
        reference = ""
        if reference_docs:
            reference = get_references(reference_docs)
        logger.info(f"reference: {reference}")
    
    return final_result+reference, image_url, reference_docs

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

        content_type = utils.get_contents_type(file_name)       
        logger.info(f"content_type: {content_type}") 

        if content_type == "image/jpeg" or content_type == "image/png":
            s3_key = f"{s3_image_prefix}/{file_name}"
        else:
            s3_key = f"{s3_prefix}/{file_name}"
        
        user_meta = {  # user-defined metadata
            "content_type": content_type,
            "model_name": model_name
        }
        
        response = s3_client.put_object(
            Bucket=s3_bucket, 
            Key=s3_key, 
            ContentType=content_type,
            Metadata = user_meta,
            Body=file_bytes            
        )
        logger.info(f"upload response: {response}")

        url = f"https://{s3_bucket}.s3.amazonaws.com/{s3_key}"
        return url
    
    except Exception as e:
        err_msg = f"Error uploading to S3: {str(e)}"
        logger.info(f"{err_msg}")
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
            logger.info(f"{err_msg}")
            continue

    return images

# load csv documents from s3
def load_csv_document(s3_file_name):
    s3r = boto3.resource("s3")
    doc = s3r.Object(s3_bucket, s3_prefix+'/'+s3_file_name)

    lines = doc.get()['Body'].read().decode('utf-8').split('\n')   # read csv per line
    logger.info(f"prelinspare: {len(lines)}")
        
    columns = lines[0].split(',')  # get columns
    #columns = ["Category", "Information"]  
    #columns_to_metadata = ["type","Source"]
    logger.info(f"columns: {columns}")
    
    docs = []
    n = 0
    for row in csv.DictReader(lines, delimiter=',',quotechar='"'):
        # print('row: ', row)
        #to_metadata = {col: row[col] for col in columns_to_metadata if col in row}
        values = {k: row[k] for k in columns if k in row}
        content = "\n".join(f"{k.strip()}: {v.strip()}" for k, v in values.items())
        doc = Document(
            page_content=content,
            metadata={
                'name': s3_file_name,
                'row': n+1,
            }
            #metadata=to_metadata
        )
        docs.append(doc)
        n = n+1
    logger.info(f"docs[0]: {docs[0]}")

    return docs

def get_summary(docs):    
    llm = get_chat()

    text = ""
    for doc in docs:
        text = text + doc
    
    if isKorean(text)==True:
        system = (
            "다음의 <article> tag안의 문장을 요약해서 500자 이내로 설명하세오."
        )
    else: 
        system = (
            "Here is pieces of article, contained in <article> tags. Write a concise summary within 500 characters."
        )
    
    human = "<article>{text}</article>"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    # print('prompt: ', prompt)
    
    chain = prompt | llm    
    try: 
        result = chain.invoke(
            {
                "text": text
            }
        )
        
        summary = result.content
        logger.info(f"esult of summarization: {summary}")
    except Exception:
        err_msg = traceback.format_exc()
        logger.info(f"error message: {err_msg}") 
        raise Exception ("Not able to request to LLM")
    
    return summary

# load documents from s3 for pdf and txt
def load_document(file_type, s3_file_name):
    s3r = boto3.resource("s3")
    doc = s3r.Object(s3_bucket, s3_prefix+'/'+s3_file_name)
    logger.info(f"s3_bucket: {s3_bucket}, s3_prefix: {s3_prefix}, s3_file_name: {s3_file_name}")
    
    contents = ""
    if file_type == 'pdf':
        contents = doc.get()['Body'].read()
        reader = PyPDF2.PdfReader(BytesIO(contents))
        
        raw_text = []
        for page in reader.pages:
            raw_text.append(page.extract_text())
        contents = '\n'.join(raw_text)    
        
    elif file_type == 'txt' or file_type == 'md':        
        contents = doc.get()['Body'].read().decode('utf-8')
        
    logger.info(f"contents: {contents}")
    new_contents = str(contents).replace("\n"," ") 
    logger.info(f"length: {len(new_contents)}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""],
        length_function = len,
    ) 
    texts = text_splitter.split_text(new_contents) 
    if texts:
        logger.info(f"exts[0]: {texts[0]}")
    
    return texts

def summary_of_code(code, mode):
    if mode == 'py':
        system = (
            "다음의 <article> tag에는 python code가 있습니다."
            "code의 전반적인 목적에 대해 설명하고, 각 함수의 기능과 역할을 자세하게 한국어 500자 이내로 설명하세요."
        )
    elif mode == 'js':
        system = (
            "다음의 <article> tag에는 node.js code가 있습니다." 
            "code의 전반적인 목적에 대해 설명하고, 각 함수의 기능과 역할을 자세하게 한국어 500자 이내로 설명하세요."
        )
    else:
        system = (
            "다음의 <article> tag에는 code가 있습니다."
            "code의 전반적인 목적에 대해 설명하고, 각 함수의 기능과 역할을 자세하게 한국어 500자 이내로 설명하세요."
        )
    
    human = "<article>{code}</article>"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    # print('prompt: ', prompt)
    
    llm = get_chat()

    chain = prompt | llm    
    try: 
        result = chain.invoke(
            {
                "code": code
            }
        )
        
        summary = result.content
        logger.info(f"result of code summarization: {summary}")
    except Exception:
        err_msg = traceback.format_exc()
        logger.info(f"error message: {err_msg}")        
        raise Exception ("Not able to request to LLM")
    
    return summary

def summary_image(img_base64, instruction):      
    llm = get_chat()

    if instruction:
        logger.info(f"instruction: {instruction}")
        query = f"{instruction}. <result> tag를 붙여주세요."
        
    else:
        query = "이미지가 의미하는 내용을 풀어서 자세히 알려주세요. markdown 포맷으로 답변을 작성합니다."
    
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
    
    for attempt in range(5):
        logger.info(f"attempt: {attempt}")
        try: 
            result = llm.invoke(messages)
            
            extracted_text = result.content
            # print('summary from an image: ', extracted_text)
            break
        except Exception:
            err_msg = traceback.format_exc()
            logger.info(f"error message: {err_msg}")                    
            raise Exception ("Not able to request to LLM")
        
    return extracted_text

def extract_text(img_base64):    
    multimodal = get_chat()
    query = "텍스트를 추출해서 markdown 포맷으로 변환하세요. <result> tag를 붙여주세요."
    
    extracted_text = ""
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
    
    for attempt in range(5):
        logger.info(f"attempt: {attempt}")
        try: 
            result = multimodal.invoke(messages)
            
            extracted_text = result.content
            # print('result of text extraction from an image: ', extracted_text)
            break
        except Exception:
            err_msg = traceback.format_exc()
            logger.info(f"error message: {err_msg}")                    
            # raise Exception ("Not able to request to LLM")
    
    logger.info(f"xtracted_text: {extracted_text}")
    if len(extracted_text)<10:
        extracted_text = "텍스트를 추출하지 못하였습니다."    

    return extracted_text

fileId = uuid.uuid4().hex
# print('fileId: ', fileId)
def get_summary_of_uploaded_file(file_name, st):
    file_type = file_name[file_name.rfind('.')+1:len(file_name)]            
    logger.info(f"file_type: {file_type}")
    
    if file_type == 'csv':
        docs = load_csv_document(file_name)
        contexts = []
        for doc in docs:
            contexts.append(doc.page_content)
        logger.info(f"contexts: {contexts}")
    
        msg = get_summary(contexts)

    elif file_type == 'pdf' or file_type == 'txt' or file_type == 'md' or file_type == 'pptx' or file_type == 'docx':
        texts = load_document(file_type, file_name)

        if len(texts):
            docs = []
            for i in range(len(texts)):
                docs.append(
                    Document(
                        page_content=texts[i],
                        metadata={
                            'name': file_name,
                            # 'page':i+1,
                            'url': path+'/'+doc_prefix+parse.quote(file_name)
                        }
                    )
                )
            logger.info(f"docs[0]: {docs[0]}") 
            logger.info(f"docs size: {len(docs)}")

            contexts = []
            for doc in docs:
                contexts.append(doc.page_content)
            logger.info(f"contexts: {contexts}")

            msg = get_summary(contexts)
        else:
            msg = "문서 로딩에 실패하였습니다."
        
    elif file_type == 'py' or file_type == 'js':
        s3r = boto3.resource("s3")
        doc = s3r.Object(s3_bucket, s3_prefix+'/'+file_name)
        
        contents = doc.get()['Body'].read().decode('utf-8')
        
        #contents = load_code(file_type, object)                
                        
        msg = summary_of_code(contents, file_type)                  
        
    elif file_type == 'png' or file_type == 'jpeg' or file_type == 'jpg':
        logger.info(f"multimodal: {file_name}")
        
        s3_client = boto3.client(
            service_name='s3',
            region_name=bedrock_region
        )             
        if debug_mode=="Enable":
            status = "이미지를 가져옵니다."
            logger.info(f"status: {status}")
            st.info(status)
            
        image_obj = s3_client.get_object(Bucket=s3_bucket, Key=s3_prefix+'/'+file_name)
        # print('image_obj: ', image_obj)
        
        image_content = image_obj['Body'].read()
        img = Image.open(BytesIO(image_content))
        
        width, height = img.size 
        logger.info(f"width: {width}, height: {height}, size: {width*height}")
        
        isResized = False
        while(width*height > 5242880):                    
            width = int(width/2)
            height = int(height/2)
            isResized = True
            logger.info(f"width: {width}, height: {height}, size: {width*height}")
        
        if isResized:
            img = img.resize((width, height))
        
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
               
        # extract text from the image
        if debug_mode=="Enable":
            status = "이미지에서 텍스트를 추출합니다."
            logger.info(f"status: {status}")
            st.info(status)
        
        text = extract_text(img_base64)
        # print('extracted text: ', text)

        if text.find('<result>') != -1:
            extracted_text = text[text.find('<result>')+8:text.find('</result>')] # remove <result> tag
            # print('extracted_text: ', extracted_text)
        else:
            extracted_text = text

        if debug_mode=="Enable":
            logger.info(f"### 추출된 텍스트\n\n{extracted_text}")
            print('status: ', status)
            st.info(status)
    
        if debug_mode=="Enable":
            status = "이미지의 내용을 분석합니다."
            logger.info(f"status: {status}")
            st.info(status)

        image_summary = summary_image(img_base64, "")
        logger.info(f"image summary: {image_summary}")
            
        if len(extracted_text) > 10:
            contents = f"## 이미지 분석\n\n{image_summary}\n\n## 추출된 텍스트\n\n{extracted_text}"
        else:
            contents = f"## 이미지 분석\n\n{image_summary}"
        logger.info(f"image content: {contents}")

        msg = contents

    global fileId
    fileId = uuid.uuid4().hex
    # print('fileId: ', fileId)

    return msg

####################### LangChain #######################
# Image Summarization
#########################################################
def get_image_summarization(object_name, prompt, st):
    # load image
    s3_client = boto3.client(
        service_name='s3',
        region_name=bedrock_region
    )

    if debug_mode=="Enable":
        status = "이미지를 가져옵니다."
        logger.info(f"status: {status}")
        st.info(status)
                
    image_obj = s3_client.get_object(Bucket=s3_bucket, Key=s3_image_prefix+'/'+object_name)
    # print('image_obj: ', image_obj)
    
    image_content = image_obj['Body'].read()
    img = Image.open(BytesIO(image_content))
    
    width, height = img.size 
    logger.info(f"width: {width}, height: {height}, size: {width*height}")
    
    isResized = False
    while(width*height > 5242880):                    
        width = int(width/2)
        height = int(height/2)
        isResized = True
        logger.info(f"width: {width}, height: {height}, size: {width*height}")
    
    if isResized:
        img = img.resize((width, height))
    
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # extract text from the image
    if debug_mode=="Enable":
        status = "이미지에서 텍스트를 추출합니다."
        logger.info(f"status: {status}")
        st.info(status)

    text = extract_text(img_base64)
    logger.info(f"extracted text: {text}")

    if text.find('<result>') != -1:
        extracted_text = text[text.find('<result>')+8:text.find('</result>')] # remove <result> tag
        # print('extracted_text: ', extracted_text)
    else:
        extracted_text = text
    
    if debug_mode=="Enable":
        status = f"### 추출된 텍스트\n\n{extracted_text}"
        logger.info(f"status: {status}")
        st.info(status)
    
    if debug_mode=="Enable":
        status = "이미지의 내용을 분석합니다."
        logger.info(f"status: {status}")
        st.info(status)

    image_summary = summary_image(img_base64, prompt)
    
    if text.find('<result>') != -1:
        image_summary = image_summary[image_summary.find('<result>')+8:image_summary.find('</result>')]
    logger.info(f"image summary: {image_summary}")
            
    if len(extracted_text) > 10:
        contents = f"## 이미지 분석\n\n{image_summary}\n\n## 추출된 텍스트\n\n{extracted_text}"
    else:
        contents = f"## 이미지 분석\n\n{image_summary}"
    logger.info(f"image contents: {contents}")

    return contents

####################### Bedrock Agent #######################
# Bedrock Agent (Multi agent collaboration)
############################################################# 

# supervisor
supervisor_agent_id = supervisor_alias_id = None
supervisor_agent_name = "supervisor"
supervisor_agent_alias_name = "latest_version"
supervisor_agent_alias_arn = ""

# collaborator
stock_agent_id = stock_agent_alias_id = None
stock_agent_name = "stock"
stock_agent_alias_name = "latest_version"
stock_agent_alias_arn = ""

search_agent_id = search_agent_alias_id = None
search_agent_name = "search"
search_agent_alias_name = "latest_version"
search_agent_alias_arn = ""

def run_bedrock_multi_agent_collaboration(text, st):
    global stock_agent_id, stock_agent_alias_id, search_agent_id, search_agent_alias_id, supervisor_agent_id, supervisor_agent_alias_id
    global stock_agent_alias_arn, search_agent_alias_arn, supervisor_agent_alias_arn
    # collaborator: stock agent
    stock_agent_id, stock_agent_alias_id, stock_agent_alias_arn = check_bedrock_multi_agent_status("COLLABORATOR", stock_agent_name, stock_agent_alias_name, stock_agent_alias_id, stock_agent_alias_arn, st)
    logger.info(f"stock_agent_id: {stock_agent_id} stock_agent_alias_id: {stock_agent_alias_id}")

    # collaborator: search agent
    search_agent_id, search_agent_alias_id, search_agent_alias_arn = check_bedrock_multi_agent_status("COLLABORATOR", search_agent_name, search_agent_alias_name, search_agent_alias_id, search_agent_alias_arn, st)
    logger.info(f"search_agent_id: {search_agent_id} search_agent_alias_id: {search_agent_alias_id}")

    # supervisor
    supervisor_agent_id, supervisor_agent_alias_id, supervisor_agent_alias_arn = check_bedrock_multi_agent_status("SUPERVISOR", supervisor_agent_name, supervisor_agent_alias_name, supervisor_alias_id, supervisor_agent_alias_arn, st)
    logger.info(f"supervisor_agent_id: {supervisor_agent_id} supervisor_agent_alias_id: {supervisor_agent_alias_id}")
    
    global sessionId
    if not userId in sessionId:
        sessionId[userId] = str(uuid.uuid4())

    result = ""
    image_url = []
    if supervisor_agent_id and supervisor_agent_alias_id:
        if debug_mode=="Enable":
            st.info('답변을 생성하고 있습니다.')

        client_runtime = boto3.client(
            service_name='bedrock-agent-runtime',
            region_name=bedrock_region
        )
        try:
            response = client_runtime.invoke_agent( 
                agentAliasId=supervisor_agent_alias_id,
                agentId=supervisor_agent_id,
                inputText=text, 
                enableTrace=True,
                sessionId=sessionId[userId], 
                memoryId='memory-'+userId
            )
            logger.info(f"response of invoke_agent(): {response}")
            
            response_stream = response['completion']

            final_result = ""    
            image_url = []
            
            for index, event in enumerate(response_stream):
                result, image_url = show_output(event, st)
                if result:
                    logger.info(f"event: {index}, result: {result}")
                    final_result = result
                    
        except Exception as e:
            if debug_mode=="Enable":
                st.error('실패하여 agent 정보를 초기화하였습니다. 재시도해주세요.')
            err_msg = traceback.format_exc()
            logger.info(f"error message: {err_msg}")
                    
    return final_result, image_url

def create_bedrock_agent_collaborator(modelId, modelName, agentName, agentAliasName, st):
    if agentName == "stock":
        functionSchema = {
            'functions': [
                {
                    'name': 'stock',
                    'description': "Retrieve accurate stock data for a given ticker.",
                    'parameters': {
                        'ticker': {
                            'description': 'the ticker to retrieve price history for. In South Korea, a ticker is a 6-digit number.',
                            'required': True,
                            'type': 'string'
                        },
                        'country': {
                            'description': 'the English country name of the stock',
                            'required': True,
                            'type': 'string'
                        }
                    },
                    'requireConfirmation': 'DISABLED'
                }
            ]
        }
    elif agentName == "search": 
        functionSchema = {
            'functions': [
                {
                    'name': 'search_internet',
                    'description': "Search general information by keyword and then return the result as a string.",
                    'parameters': {
                        'keyword': {
                            'description': 'search keyword',
                            'required': True,
                            'type': 'string'
                        }
                    },
                    'requireConfirmation': 'DISABLED'
                }
            ]
        }

    # create collaborator agent
    if debug_mode=="Enable":
        st.info(f"Collaborator Agent인 {agentName}를 생성합니다. 사용 모델은 {modelName}입니다.")

    agent_instruction = (
        "당신의 이름은 서연이고, 질문에 친근한 방식으로 대답하도록 설계된 대화형 AI입니다. "
        "상황에 맞는 구체적인 세부 정보를 충분히 제공합니다. "
        "모르는 질문을 받으면 솔직히 모른다고 말합니다. "
    )
    logger.info(f"modelId: {modelId}")

    response = client.create_agent(
        agentResourceRoleArn=agent_role_arn,
        instruction=agent_instruction,
        foundationModel=modelId,
        description=f"Collaborator Agent인 {agentName}입니다. 사용 모델은 {modelName}입니다.",
        agentName=agentName,
        idleSessionTTLInSeconds=600
    )
    logger.info(f"response of create_bedrock_agent_collaborator(): {response}")

    agentId = response['agent']['agentId']
    logger.info(f"agentId: {agentId}")
    time.sleep(5)   

    # create action group    
    create_action_group(agentId, action_group_name_for_multi_agent, lambda_tools_arn, functionSchema, st)     

    if agentName == "stock":
        create_action_group_for_code_interpreter(agentId, st)
    
    # preparing
    if debug_mode=="Enable":
        st.info('Agent를 사용할 수 있도록 "Prepare"로 설정합니다.')    
    prepare_agent(agentId)
    
    # deploy
    if debug_mode=="Enable":
        st.info(f'{agentName}을 {agentAliasName}로 배포합니다.')    
    agentAliasId, agentAliasArn = deploy_agent(agentId, agentAliasName)
    time.sleep(3) 

    logger.info(f"agentName: {agentName}, agentId: {agentId}, agentAliasId: {agentAliasId}, agentAliasArn: {agentAliasArn}")

    return agentId, agentAliasId, agentAliasArn

def create_bedrock_agent_supervisor(modelId, modelName, agentName, agentAliasName, st):
    # create supervisor agent
    if debug_mode=="Enable":
        st.info(f"Supervisor Agent인 {agentName}를 생성합니다. 사용 모델은 {modelName}입니다.")

    agent_instruction = (
        "당신의 이름은 서연이고, 질문에 친근한 방식으로 대답하도록 설계된 대화형 AI입니다. "
        "상황에 맞는 구체적인 세부 정보를 충분히 제공합니다. "
        "모르는 질문을 받으면 솔직히 모른다고 말합니다. "
    )
    logger.info(f"modelId: {modelId}")

    response = client.create_agent(
        agentCollaboration = 'SUPERVISOR', # SUPERVISOR_ROUTER
        orchestrationType = 'DEFAULT',
        agentName=agentName,
        agentResourceRoleArn=agent_role_arn,
        instruction=agent_instruction,
        foundationModel=modelId,
        description=f"Supervisor Agent인 {agentName}입니다. 사용 모델은 {modelName}입니다.",
        idleSessionTTLInSeconds=600
    )
    logger.info(f"response of create_agent(): {response}")

    agentId = response['agent']['agentId']
    logger.info(f"Supervisor agentId: {agentId}")
    time.sleep(3)

    # add code interpreter action group
    create_action_group_for_code_interpreter(agentId, st)
                
    # add stock agent
    logger.info(f"stock_agent_alias_arn: {stock_agent_alias_arn}")

    response = client.associate_agent_collaborator(
        agentDescriptor={
            'aliasArn': stock_agent_alias_arn
        },
        agentId=agentId,
        agentVersion='DRAFT',
        collaborationInstruction=f"{stock_agent_name} retrieves accurate stock data for a given ticker.",
        collaboratorName=stock_agent_name
    )
    logger.info(f"response of associate_agent_collaborator(): {response}")
    
    # add search agent
    logger.info(f"search_agent_alias_arn: {search_agent_alias_arn}")

    response = client.associate_agent_collaborator(
        agentDescriptor={
            'aliasArn': search_agent_alias_arn
        },
        agentId=agentId,
        agentVersion='DRAFT',
        collaborationInstruction=f"{search_agent_name} searchs general information by keyword and then return the result as a string.",
        collaboratorName=search_agent_name
    )
    logger.info(f"response of associate_agent_collaborator(): {response}")
    time.sleep(3)

    # preparing
    if debug_mode=="Enable":
        st.info('Agent를 사용할 수 있도록 "Prepare"로 설정합니다.')    
    prepare_agent(agentId)
    time.sleep(3)
    
    # deploy
    if debug_mode=="Enable":
        st.info(f'{agentName}을 {agentAliasName}로 배포합니다.')    
    agentAliasId, agentAliasArn = deploy_agent(agentId, agentAliasName)    
    time.sleep(3)

    return agentId, agentAliasId, agentAliasArn

def check_bedrock_multi_agent_status(agentType, agentName, agentAliasName, agentAliasId, agentAliasArn, st):
    agentId = retrieve_agent_id(agentName)  
    
    # create collaborator agent if no agent
    if not agentId and agentType=="COLLABORATOR":
        agentId, agentAliasId, agentAliasArn = create_bedrock_agent_collaborator(model_id, model_name, agentName, agentAliasName, st)           
    if not agentId and agentType=="SUPERVISOR":
        agentId, agentAliasId, agentAliasArn = create_bedrock_agent_supervisor(model_id, model_name, agentName, agentAliasName, st)     
    logger.info(f"agentId: {agentId}, agentAliasId: {agentAliasId}, agentAliasArn: {agentAliasArn}")

    if not agentAliasId and agentId:
        if debug_mode=="Enable":
            st.info(f"{agentName}의 alias를 검색합니다.")

        # retrieve agent alias
        response_agent_alias = client.list_agent_aliases(
            agentId = agentId,
            maxResults=10
        )
        logger.info(f"response of list_agent_aliases(): {response_agent_alias}")

        for summary in response_agent_alias["agentAliasSummaries"]:
            if summary["agentAliasName"] == agentAliasName:
                agentAliasId = summary["agentAliasId"]
                logger.info(f"agentAliasId: {agentAliasId}")

                if not agentAliasArn:
                    response = client.get_agent_alias(
                        agentAliasId=agentAliasId,
                        agentId=agentId
                    )
                    logger.info(f"response of get_agent_alias(): {response}")

                    agentAliasArn = response["agentAlias"]["agentAliasArn"]
                    logger.info(f"agentAliasArn: {agentAliasArn}")

                logger.info(f"agentAliasStatus: {summary['agentAliasStatus']}")
                if not summary["agentAliasStatus"] == "PREPARED":
                    if debug_mode=="Enable":
                        st.info('Agent를 사용할 수 있도록 "Prepare"로 다시 설정합니다.')
                    prepare_agent(agentId)
                break
        
        # creat agent alias if no alias
        if not agentAliasId:
            if debug_mode=="Enable":
                st.info('Agent의 alias를 생성합니다.')

            response = client.create_agent_alias(
                agentAliasName=agentAliasName,
                agentId=agentId,
                description='the lastest deployment'
            )
            logger.info(f"response of create_agent_alias(): {response}")

            agentAliasId = response['agentAlias']['agentAliasId']
            agentAliasArn = response['agentAlias']['agentAliasArn']
            logger.info(f"agentAliasId: {agentAliasId}, agentAliasArn: {agentAliasArn}")
            time.sleep(5) # delay 5 seconds

            agentAliasId, agentAliasArn = deploy_agent(agentId, agentAliasName)
        
    return agentId, agentAliasId, agentAliasArn

