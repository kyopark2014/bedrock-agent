# Bedrock Agent

<p align="left">
    <a href="https://hits.seeyoufarm.com"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fkyopark2014%2Fbedrock-agent&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com"/></a>
    <img alt="License" src="https://img.shields.io/badge/LICENSE-MIT-green">
</p>


여기에서는 Knowledge Base로 RAG를 구성하고 이를 바탕으로 bedrock agent의 사용해 봅니다.

## 전체 Architecture

<img width="800" alt="image" src="https://github.com/user-attachments/assets/aef453d3-c7d0-44d3-af93-24c9031bb7ec" />

## 상세 구현

### RAG


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

lambda-tools를 압축해서 올리는 명령어는 아래와 같습니다. 압축후에 my_deployment_package.zip을 console에서 업로드합니다.

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

