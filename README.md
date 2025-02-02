# Bedrock Agent

<p align="left">
    <a href="https://hits.seeyoufarm.com"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fkyopark2014%2Fbedrock-agent&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com"/></a>
    <img alt="License" src="https://img.shields.io/badge/LICENSE-MIT-green">
</p>


여기에서는 Knowledge Base로 RAG를 구성하고 bedrock agent를 활용하는 방법을 설명합니다. Bedrock agent의 LLM 모델로는 Anthropic의 Claude와 Amazon의 Nova를 활용합니다. 

## 전체 Architecture

여기에서는 Knowledge Base를 이용해 지식저장소를 활용합니다.

<img width="800" alt="image" src="https://github.com/user-attachments/assets/aef453d3-c7d0-44d3-af93-24c9031bb7ec" />

## 상세 구현

### 일반적인 대화

일반적인 대화에서는 프롬프트의 chatbot의 이름을 지정하고 원하는 동작을 수행하도록 요청할 수 있습니다. 또한 아래와 같이 history를 이용해 이전 대화내용을 참조하여 답변하도록 합니다. 결과는 stream으로 전달되어 streamlit으로 표시됩니다.

```python
def general_conversation(query):
    chat = get_chat()
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
    chain = prompt | chat | StrOutputParser()
    stream = chain.stream(
        {
            "history": history,
            "input": query,
        }
    )          
    return stream
```

### RAG

RAG은 retrive, grade, generation의 단계로 수행됩니다. Knowledge Base를 이용해 관련된 문서를 가져오는 retrieve 동작을 수행합니다. 문서 원본을 확인할 수 있도록 파일의 url은 cloudfront의 도메인을 기준으로 제공합니다. 문서의 조회는 LangChain의 [AmazonKnowledgeBasesRetriever](https://api.python.langchain.com/en/latest/community/retrievers/langchain_community.retrievers.bedrock.AmazonKnowledgeBasesRetriever.html)을 이용하여 numberOfResults만큼 가져옵니다. 이때 overrideSearchType로 'HYBRID'와 'SEMANTIC"을 선택할 수 있습니다. LangChain을 사용하지 않을 경우에 boto3의 [retrieve_and_generate](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-agent-runtime/client/retrieve_and_generate.html#)을 이용해 유사하게 구현이 가능합니다. 

```python
def retrieve_documents_from_knowledge_base(query, top_k):
    relevant_docs = []
    retriever = AmazonKnowledgeBasesRetriever(
        knowledge_base_id=knowledge_base_id, 
        retrieval_config={"vectorSearchConfiguration": {
            "numberOfResults": top_k,
            "overrideSearchType": "HYBRID"   # SEMANTIC
        }},
        region_name=bedrock_region
    )
    
    documents = retriever.invoke(query)    
    relevant_docs = []
    for doc in documents:
        content = ""
        if doc.page_content:
            content = doc.page_content        
        score = doc.metadata["score"]        
        if "s3Location" in doc.metadata["location"]:
            link = doc.metadata["location"]["s3Location"]["uri"] if doc.metadata["location"]["s3Location"]["uri"] is not None else ""
            pos = link.find(f"/{doc_prefix}")
            name = link[pos+len(doc_prefix)+1:]
            encoded_name = parse.quote(name)
            link = f"{path}/{doc_prefix}{encoded_name}"
        elif "webLocation" in doc.metadata["location"]:
            link = doc.metadata["location"]["webLocation"]["url"] if doc.metadata["location"]["webLocation"]["url"] is not None else ""
            name = "WEB"
        url = link
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
```

적절한 문서를 선택하기 위하여 grade 동작을 수행합니다. 아래와 같이 retieve에서 얻어진 관련된 문서들이 실제 관련이 있는데 확인합니다. 이때 확인된 문서들만  filtered_docs로 정리합니다.

```python
def grade_documents(question, documents):
    filtered_docs = []
    chat = get_chat()
    retrieval_grader = get_retrieval_grader(chat)
    for i, doc in enumerate(documents):
        score = retrieval_grader.invoke({"question": question, "document": doc.page_content})
        grade = score.binary_score
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(doc)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue
    return filtered_docs
```

문서가 실제 관련이 있는지는 아래와 같이 prompt를 이용합니다. 이때 "yes", "no"와 같은 결과를 추출하기 위하여 structured_output을 활용하였습니다.

```python
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
```

관련된 문서를 가지고 답변을 생성합니다.

```python
chat = get_chat()
system = (
    "당신의 이름은 서연이고, 질문에 대해 친절하게 답변하는 사려깊은 인공지능 도우미입니다."
    "다음의 Reference texts을 이용하여 user의 질문에 답변합니다."
    "모르는 질문을 받으면 솔직히 모른다고 말합니다."
    "답변의 이유를 풀어서 명확하게 설명합니다."
)
human = (
    "Question: {question}"

    "Reference texts: "
    "{context}"
) 
 prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
rag_chain = prompt | chat
result = rag_chain.invoke(
    {
        "question": text,
        "context": relevant_context                
    }
)
msg = result.content        
```

### Bedrock Agent

Agent를 위해서는 [cdk-bedrock-agent-stack.ts](./cdk-bedrock-agent/lib/cdk-bedrock-agent-stack.ts)와 같이 Bedrock에 대한 invoke, retrieve, inference를 허용하도록 하여야 합니다.

```python
const agent_role = new iam.Role(this,  `role-agent-for-${projectName}`, {
  roleName: `role-agent-for-${projectName}-${region}`,
  assumedBy: new iam.CompositePrincipal(
    new iam.ServicePrincipal("bedrock.amazonaws.com")
  )
});

const agentInvokePolicy = new iam.PolicyStatement({ 
  effect: iam.Effect.ALLOW,
  resources: [
    `arn:aws:bedrock:*::foundation-model/*`
  ],
  actions: [
    "bedrock:InvokeModel"
  ],
});        
agent_role.attachInlinePolicy( 
  new iam.Policy(this, `agent-invoke-policy-for-${projectName}`, {
    statements: [agentInvokePolicy],
  }),
);  

const bedrockRetrievePolicy = new iam.PolicyStatement({ 
  effect: iam.Effect.ALLOW,
  resources: [
    `arn:aws:bedrock:${region}:${accountId}:knowledge-base/*`
  ],
  actions: [
    "bedrock:Retrieve"
  ],
});        
agent_role.attachInlinePolicy( 
  new iam.Policy(this, `bedrock-retrieve-policy-for-${projectName}`, {
    statements: [bedrockRetrievePolicy],
  }),
);  

const agentInferencePolicy = new iam.PolicyStatement({ 
  effect: iam.Effect.ALLOW,
  resources: [
    `arn:aws:bedrock:${region}:${accountId}:inference-profile/*`,
    `arn:aws:bedrock:*::foundation-model/*`
  ],
  actions: [
    "bedrock:InvokeModel",
    "bedrock:GetInferenceProfile",
    "bedrock:GetFoundationModel"
  ],
});        
agent_role.attachInlinePolicy( 
  new iam.Policy(this, `agent-inference-policy-for-${projectName}`, {
    statements: [agentInferencePolicy],
  }),
);
```

Bedrock agent는 console에서 생성할 수도 있지만 아래와 같이 boto3의 [create_agent](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-agent/client/create_agent.html)을 이용해 생성할 수 있습니다. 이때 agent의 instruction을 지정하고 agent role을 활용합니다. 생성된 agentId는 이후 agent에 추가 설정을 하거나 실행할때 활용됩니다.

```python
client = boto3.client(
    service_name='bedrock-agent',
    region_name=bedrock_region
)  
agent_instruction = (
    "당신의 이름은 서연이고, 질문에 친근한 방식으로 대답하도록 설계된 대화형 AI입니다. "
    "상황에 맞는 구체적인 세부 정보를 충분히 제공합니다. "
    "모르는 질문을 받으면 솔직히 모른다고 말합니다. "
)
response = client.create_agent(
    agentResourceRoleArn=agent_role_arn,
    instruction=agent_instruction,
    foundationModel=modelId,
    description=f"Bedrock Agent (Knowledge Base) 입니다. 사용 모델은 {modelName}입니다.",
    agentName=agentName,
    idleSessionTTLInSeconds=600
)
agentId = response['agent']['agentId']
```

Bedrock agent에서 실행할 tool들은 action group에서 정의합니다. Action group이 이미 있는지를 [list_agent_action_groups](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-agent/client/list_agent_action_groups.html)로 확인하고, [create_agent_action_group](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-agent/client/create_agent_action_group.html)을 이용해 action group을 생성할 수 있습니다.

```python
response = client.list_agent_action_groups(
    agentId=agentId,
    agentVersion='DRAFT',
    maxResults=10
)
actionGroupSummaries = response['actionGroupSummaries']

isExist = False
for actionGroup in actionGroupSummaries:
    if actionGroup['actionGroupId'] == actionGroupName:
        isExist = True
        break
if not isExist:
    response = client.create_agent_action_group(
        actionGroupName=actionGroupName,
        actionGroupState='ENABLED',
        agentId=agentId,
        agentVersion='DRAFT',
        description=f"Action Group의 이름은 {actionGroupName} 입니다.",
        actionGroupExecutor={'lambda': lambda_tools_arn},
        functionSchema={
            'functions': [
                {
                    'name': 'get_book_list',
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
                    'name': 'get_current_time',
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
                    'name': 'get_weather_info',
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
                    'name': 'search_by_tavily',
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
                    'name': 'search_by_knowledge_base',
                    'description': "Search technical information by keyword and then return the result as a string.",
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
        },            
    )
```

생성된 Bedrock agent에 RAG를 직접 연결할 때에는 아래와 같이 [associate_agent_knowledge_base](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-agent/client/associate_agent_knowledge_base.html)을 이용하여 연결합니다.

```python
rag_prompt = (
    "당신의 이름은 서연이고, 질문에 대해 친절하게 답변하는 사려깊은 인공지능 도우미입니다."
    "다음의 Reference texts을 이용하여 user의 질문에 답변합니다."
    "모르는 질문을 받으면 솔직히 모른다고 말합니다."
    "답변의 이유를 풀어서 명확하게 설명합니다."
)
response = client.associate_agent_knowledge_base(
    agentId=agentId,
    agentVersion='DRAFT',
    description=rag_prompt,
    knowledgeBaseId=knowledge_base_id,
    knowledgeBaseState='ENABLED'
)
print(f'response of associate_agent_knowledge_base(): {response}')
```

Bedrock agent를 이용하려면 실행전에 prepared 상태이어야 합니다. 따라서 설정 후에는 아래처럼 prepare 상태를 변경합니다. 

```python
response = client.prepare_agent(
    agentId=agentId
)
print('response of prepare_agent(): ', response)      
```

Bedrock agent를 사용하기 위해서는 배포를 하여야 하는데, 여기서는 원할한 데모를 위해 기존 배포가 있다지 확인해서 있다면 지우고 새로 생성합니다. 기존 배포의 확인은 [list_agent_aliases](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-agent/client/list_agent_aliases.html), 삭제는 [delete_agent_alias](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-agent/client/delete_agent_alias.html), 생성은 [create_agent_alias](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-agent/client/create_agent_alias.html)을 이용합니다.

```python
# retrieve agent alias
response_agent_alias = client.list_agent_aliases(
    agentId = agentId,
    maxResults=10
)
for summary in response_agent_alias["agentAliasSummaries"]:
    if summary["agentAliasName"] == agentAliasName:
        agentAliasId = summary["agentAliasId"]
        break
if agentAliasId:
    response = client.delete_agent_alias(
        agentAliasId=agentAliasId,
        agentId=agentId
    )            

# create agent alias 
response = client.create_agent_alias(
    agentAliasName=agentAliasName,
    agentId=agentId,
    description='the lastest deployment'
)
agentAliasId = response['agentAlias']['agentAliasId']
```

Bedrock agent는 "bedrock-agent-runtime"을 이용하여 [invoke_agent](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-agent-runtime/client/invoke_agent.html)로 실행합니다.

```python
client_runtime = boto3.client(
    service_name='bedrock-agent-runtime',
    region_name=bedrock_region
)
response = client_runtime.invoke_agent( 
    agentAliasId=agentAliasId,
    agentId=agentId,
    inputText=text, 
    enableTrace=True,
    sessionId=sessionId[userId], 
    memoryId='memory-'+userId,
    sessionState=sessionState
)
event_stream = response['completion']
```

결과는 stream으로 얻을수 있습니다. "enableTrace"를 이용해 중간 결과를 화면에 표시할 수 있습니다. 최종 결과는 chunk난 trace의 "observation"의 "finalResponse"로 알 수 있습니다.

```python
for index, event in enumerate(event_stream):
    # Handle text chunks
    if "chunk" in event:
        chunk = event["chunk"]
        if "bytes" in chunk:
            text = chunk["bytes"].decode("utf-8")
            stream_result += text
    # Handle file outputs
    if "files" in event:
        files = event["files"]["files"]
        for file in files:
            st.image(file["bytes"], caption=file["name"])
    # Check trace
    if "trace" in event:
        if ("trace" in event["trace"] and "orchestrationTrace" in event["trace"]["trace"]):
            trace_event = event["trace"]["trace"]["orchestrationTrace"]
            if "rationale" in trace_event:
                trace_text = trace_event["rationale"]["text"]
                st.info(f"rationale: {trace_text}")
            if "modelInvocationInput" in trace_event:
                if "text" in trace_event["modelInvocationInput"]:
                    trace_text = trace_event["modelInvocationInput"]["text"]
                    print("trace_text: ", trace_text)
                if "rawResponse" in trace_event["modelInvocationInput"]:
                    rawResponse = trace_event["modelInvocationInput"]["rawResponse"]                        
                    print("rawResponse: ", rawResponse)
            if "modelInvocationOutput" in trace_event:
                if "rawResponse" in trace_event["modelInvocationOutput"]:
                    trace_text = trace_event["modelInvocationOutput"]["rawResponse"]["content"]
                    print("trace_text: ", trace_text)
            if "invocationInput" in trace_event:
                if "codeInterpreterInvocationInput" in trace_event["invocationInput"]:
                    trace_code = trace_event["invocationInput"]["codeInterpreterInvocationInput"]["code"]
                    print("trace_code: ", trace_code)
                if "knowledgeBaseLookupInput" in trace_event["invocationInput"]:
                    trace_text = trace_event["invocationInput"]["knowledgeBaseLookupInput"]["text"]
                    st.info(f"RAG를 검색합니다. 검색어: {trace_text}")
                if "actionGroupInvocationInput" in trace_event["invocationInput"]:
                    trace_function = trace_event["invocationInput"]["actionGroupInvocationInput"]["function"]
                    st.info(f"actionGroupInvocation: {trace_function}")
            if "observation" in trace_event:
                if "finalResponse" in trace_event["observation"]:
                    trace_resp = trace_event["observation"]["finalResponse"]["text"]
                    final_result = trace_resp
                if ("codeInterpreterInvocationOutput" in trace_event["observation"]):
                    if "executionOutput" in trace_event["observation"]["codeInterpreterInvocationOutput"]:
                        trace_resp = trace_event["observation"]["codeInterpreterInvocationOutput"]["executionOutput"]
                        st.info(f"observation: {trace_resp}")
                    if "executionError" in trace_event["observation"]["codeInterpreterInvocationOutput"]:
                        trace_resp = trace_event["observation"]["codeInterpreterInvocationOutput"]["executionError"]
                        if "image_url" in trace_resp:
                            print("got image")
                            image_url = trace_resp["image_url"]
                            st.image(image_url)
                if "knowledgeBaseLookupOutput" in trace_event["observation"]:
                    if "retrievedReferences" in trace_event["observation"]["knowledgeBaseLookupOutput"]:
                        references = trace_event["observation"]["knowledgeBaseLookupOutput"]["retrievedReferences"]
                        st.info(f"{len(references)}개의 문서가 검색되었습니다.")
                if "actionGroupInvocationOutput" in trace_event["observation"]:
                    trace_resp = trace_event["observation"]["actionGroupInvocationOutput"]["text"]
                    st.info(f"actionGroupInvocationOutput: {trace_resp}")
        elif "guardrailTrace" in event["trace"]["trace"]:
            guardrail_trace = event["trace"]["trace"]["guardrailTrace"]
            if "inputAssessments" in guardrail_trace:
                assessments = guardrail_trace["inputAssessments"]
                for assessment in assessments:
                    if "contentPolicy" in assessment:
                        filters = assessment["contentPolicy"]["filters"]
                        for filter in filters:
                            if filter["action"] == "BLOCKED":
                                st.error(f"Guardrail blocked {filter['type']} confidence: {filter['confidence']}")
                    if "topicPolicy" in assessment:
                        topics = assessment["topicPolicy"]["topics"]
                        for topic in topics:
                            if topic["action"] == "BLOCKED":
                                st.error(f"Guardrail blocked topic {topic['name']}")            
```

### Agent에서 사용할 Tool의 구현

Action group에 정의된 Tool들은 Lambda를 이용해 실행됩니다. [cdk-bedrock-agent-stack.ts](./cdk-bedrock-agent/lib/cdk-bedrock-agent-stack.ts)에서는 아래와 같이 tools을 위한 lambda를 정의합니다. 이때, lambda는 "lambda:InvokeFunction"에 대한 권한을 가지고 있어야 합니다.

```java
const lambdaTools = new lambda.DockerImageFunction(this, `lambda-tools-for-${projectName}`, {
  description: 'action group - tools',
  functionName: `lambda-tools-for-${projectName}`,
  code: lambda.DockerImageCode.fromImageAsset(path.join(__dirname, '../../lambda-tools')),
  timeout: cdk.Duration.seconds(60),
  environment: {
    bedrock_region: String(region),
    projectName: projectName,
    "sharing_url": 'https://'+distribution_sharing.domainName,
  }
});         
lambdaTools.grantInvoke(new cdk.aws_iam.ServicePrincipal("bedrock.amazonaws.com"));
```

Bedrock agent가 Tool을 선택하면 event로 전달됩니다. 이때 event에서 "function"으로 tool의 이름을 확인하고, "parameters"의 "value"로 tool을 실행시키고 결과를 리턴합니다. 아래의 코드는 [lambda_function.py](./lambda-tools/lambda_function.py)을 참조합니다.

```python
def lambda_handler(event, context):
    agent = event['agent']
    actionGroup = event['actionGroup']
    function = event['function']
    parameters = event.get('parameters',[])
    name = parameters[0]['name']
    value = parameters[0]['value']
    
    if function == 'get_current_time':
        output = get_current_time(value)        
    elif function == 'get_book_list':
        output = get_book_list(value)            
    elif function == 'get_weather_info':        
        output = get_weather_info(value)
    elif function == 'search_by_tavily':
        output = search_by_tavily(value)
    elif function == 'search_by_knowledge_base':
        output = search_by_knowledge_base(value)

    responseBody =  {
        "TEXT": {
            "body": output
        }
    }
    action_response = {
        'actionGroup': actionGroup,
        'function': function,
        'functionResponse': {
            'responseBody': responseBody
        }
    }
    response = {
        'response': action_response, 
        'messageVersion': event['messageVersion']
    }
    return response
```

### 활용 방법

EC2는 Private Subnet에 있으므로 SSL로 접속할 수 없습니다. 따라서, [Console-EC2](https://us-west-2.console.aws.amazon.com/ec2/home?region=us-west-2#Instances:)에 접속하여 "app-for-bedrock-agent"를 선택한 후에 Connect에서 sesseion manager를 선택하여 접속합니다. 

Github에서 app에 대한 코드를 업데이트 하였다면, session manager에 접속하여 아래 명령어로 업데이트 합니다. 

```text
sudo runuser -l ec2-user -c 'cd /home/ec2-user/bedrock-agent && git pull'
```

Streamlit의 재시작이 필요하다면 아래 명령어로 service를 stop/start 시키고 동작을 확인할 수 있습니다.

```text
sudo systemctl stop streamlit
sudo systemctl start streamlit
sudo systemctl status streamlit -l
```

Local에서 디버깅을 빠르게 진행하고 싶다면 [Local에서 실행하기](https://github.com/kyopark2014/llm-streamlit/blob/main/deployment.md#local%EC%97%90%EC%84%9C-%EC%8B%A4%ED%96%89%ED%95%98%EA%B8%B0)에 따라서 Local에 필요한 패키지와 환경변수를 업데이트 합니다. 이후 아래 명령어서 실행합니다.

```text
streamlit run application/app.py
```

EC2에서 debug을 하면서 개발할때 사용하는 명령어입니다.

먼저, 시스템에 등록된 streamlit을 종료합니다.

```text
sudo systemctl stop streamlit
```

이후 EC2를 session manager를 이용해 접속한 이후에 아래 명령어를 이용해 실행하면 로그를 보면서 수정을 할 수 있습니다. 

```text
sudo runuser -l ec2-user -c "/home/ec2-user/.local/bin/streamlit run /home/ec2-user/bedrock-agent/application/app.py"
```

참고로 lambda-tools로 docker image가 아닌 코드를 압축해서 올리는 경우에 참조할 명령어는 아래와 같습니다. 압축후에 my_deployment_package.zip을 console에서 업로드합니다.

```text
cd lambda-tools/
pip install --target ./package requests beautifulsoup4 # package 설치
cd package && zip -r ../my_deployment_package.zip .
cd .. && zip my_deployment_package.zip lambda_function.py info.py # add lambda_function.py
```




## 직접 실습 해보기

### 사전 준비 사항

이 솔루션을 사용하기 위해서는 사전에 아래와 같은 준비가 되어야 합니다.

- [AWS Account 생성](https://repost.aws/ko/knowledge-center/create-and-activate-aws-account)에 따라 계정을 준비합니다.

### CDK를 이용한 인프라 설치

본 실습에서는 us-west-2 리전을 사용합니다. [인프라 설치](./deployment.md)에 따라 CDK로 인프라 설치를 진행합니다. 

## 실행결과


메뉴에서 "Agent"를 선택하고 "여행과 관련된 책 추천해줘."라고 입력한 후에 결과를 확인합니다. 아래와 같이 action group에서 "get_book_list"를 호출하여 얻은 값을 가지고 답변을 생성하였습니다.


<img src="https://github.com/user-attachments/assets/b4966b10-1799-4552-8b1e-48bd108dd904" width="600">

Bedrock agetn의 code interpreter의 기능을 테스트하기 위하여 "code interpreter"를 enable하고 [주식 CSV 파일](./contents/stock_prices.csv)파일을 업로드합니다. 이후 "가장 변화량이 큰 주식의 지난 1년간의 변화량을 일자별로 표시하는 그래프를 그려주세요."라고 입력합니다. 이때의 결과는 아래와 같습니다.

![image](https://github.com/user-attachments/assets/2fb75f3f-c0fa-4bb4-98b6-fe933ca67152)


## 리소스 정리하기 

더이상 인프라를 사용하지 않는 경우에 아래처럼 모든 리소스를 삭제할 수 있습니다. 아래의 명령어로 전체 삭제를 합니다.

```text
cd ~/bedrock-agent/cdk-bedrock-agent && cdk destroy --all
```


## References

[generative-ai-cdk-constructs](https://awslabs.github.io/generative-ai-cdk-constructs/src/cdk-lib/bedrock/#knowledge-bases)

[On AWS CDK and Agents for Amazon Bedrock](https://medium.com/@micheldirk/aws-cdk-and-agents-for-amazon-bedrock-e313be7543fe)

[Bedrock Agents-Based AI AppBuilder Assistant with Boto3 SDK](https://github.com/aws-samples/application-builder-assistant-using-bedrock-agents-and-multiple-knowledge-bases/blob/main/ai_appbuilder_assistant/BedrockAgents_AI_AppBuilder_Assistant.ipynb)

[Agents for Amazon Bedrock - create agent](https://github.com/aws-samples/amazon-nova-samples/blob/main/multimodal-understanding/workshop/4.1_create_agent.ipynb)

[AWS re:Invent 2024 - Building an AWS solutions architect agentic app with Amazon Bedrock (DEV331)](https://www.youtube.com/watch?v=XPHOybnXCd4&t=1589s)

[Building Agentic Workflows on AWS](https://catalog.workshops.aws/building-agentic-workflows/en-US)

[Building an AWS Solutions Architect Agentic App with Amazon Bedrock](https://github.com/build-on-aws/agentic-workshop/tree/main/reinvent_2024_agentic)

[종속 항목이 있는 .zip 배포 패키지 생성](https://docs.aws.amazon.com/ko_kr/lambda/latest/dg/python-package.html#python-package-create-dependencies)

