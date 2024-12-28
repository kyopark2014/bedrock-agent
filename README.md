# Bedrock Agent

여기에서는 Knowledge Base로 RAG를 구성하고 이를 바탕으로 bedrock agent의 사용해 봅니다.

## 설치

## 실행 

### 일상적인 대화

### Agentic Workflow

### RAG

### Agent



## 직접 실습 해보기

### 사전 준비 사항

이 솔루션을 사용하기 위해서는 사전에 아래와 같은 준비가 되어야 합니다.

- [AWS Account 생성](https://repost.aws/ko/knowledge-center/create-and-activate-aws-account)에 따라 계정을 준비합니다.

### CDK를 이용한 인프라 설치

본 실습에서는 us-west-2 리전을 사용합니다. [인프라 설치](./deployment.md)에 따라 CDK로 인프라 설치를 진행합니다. 

## 실행결과






## 리소스 정리하기 

더이상 인프라를 사용하지 않는 경우에 아래처럼 모든 리소스를 삭제할 수 있습니다. 아래의 명령어로 전체 삭제를 합니다.

```text
cd ~/bedrock-agent/cdk-bedrock-agent && cdk destroy --all
```
