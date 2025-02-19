import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as elbv2 from 'aws-cdk-lib/aws-elasticloadbalancingv2';
import * as elbv2_tg from 'aws-cdk-lib/aws-elasticloadbalancingv2-targets'
import * as secretsmanager from 'aws-cdk-lib/aws-secretsmanager';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as opensearchserverless from 'aws-cdk-lib/aws-opensearchserverless';
import * as cloudFront from 'aws-cdk-lib/aws-cloudfront';
import * as origins from 'aws-cdk-lib/aws-cloudfront-origins';
import * as lambda from "aws-cdk-lib/aws-lambda";
import * as path from "path";

const projectName = `bedrock-agent`; 
const region = process.env.CDK_DEFAULT_REGION;    
const accountId = process.env.CDK_DEFAULT_ACCOUNT;
const targetPort = 8080;
const bucketName = `storage-for-${projectName}-${accountId}-${region}`; 
const vectorIndexName = projectName
const knowledge_base_name = projectName;

const enableHybridSearch = 'true';

export class CdkBedrockAgentStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // Knowledge Base Role
    const knowledge_base_role = new iam.Role(this,  `role-knowledge-base-for-${projectName}`, {
      roleName: `role-knowledge-base-for-${projectName}-${region}`,
      assumedBy: new iam.CompositePrincipal(
        new iam.ServicePrincipal("bedrock.amazonaws.com")
      )
    });
    
    const bedrockInvokePolicy = new iam.PolicyStatement({ 
      effect: iam.Effect.ALLOW,
      resources: [
        `arn:aws:bedrock:us-west-2::foundation-model/*`,
        `arn:aws:bedrock:us-east-1::foundation-model/*`,
        `arn:aws:bedrock:us-east-2::foundation-model/*`
      ],
      // resources: ['*'],
      actions: [
        "bedrock:InvokeModel", 
        "bedrock:Retrieve", 
        "bedrock:InvokeModelEndpoint", 
        "bedrock:InvokeModelEndpointAsync",        
      ],
    });        
    knowledge_base_role.attachInlinePolicy( 
      new iam.Policy(this, `bedrock-invoke-policy-for-${projectName}`, {
        statements: [bedrockInvokePolicy],
      }),
    );  
    
    const bedrockKnowledgeBaseS3Policy = new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      resources: ['*'],
      actions: [
        "s3:GetBucketLocation",
        "s3:GetObject",
        "s3:ListBucket",
        "s3:ListBucketMultipartUploads",
        "s3:ListMultipartUploadParts",
        "s3:AbortMultipartUpload",
        "s3:CreateBucket",
        "s3:PutObject",
        "s3:PutBucketLogging",
        "s3:PutBucketVersioning",
        "s3:PutBucketNotification",
      ],
    });
    knowledge_base_role.attachInlinePolicy( 
      new iam.Policy(this, `knowledge-base-s3-policy-for-${projectName}`, {
        statements: [bedrockKnowledgeBaseS3Policy],
      }),
    );  
    
    const knowledgeBaseOpenSearchPolicy = new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      resources: ['*'],
      actions: ["aoss:APIAccessAll"],
    });
    knowledge_base_role.attachInlinePolicy( 
      new iam.Policy(this, `bedrock-agent-opensearch-policy-for-${projectName}`, {
        statements: [knowledgeBaseOpenSearchPolicy],
      }),
    );  

    const knowledgeBaseBedrockPolicy = new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      resources: ['*'],
      actions: ["bedrock:*"],
    });
    knowledge_base_role.attachInlinePolicy( 
      new iam.Policy(this, `bedrock-agent-bedrock-policy-for-${projectName}`, {
        statements: [knowledgeBaseBedrockPolicy],
      }),
    );  

    // OpenSearch Serverless
    const collectionName = vectorIndexName
    const OpenSearchCollection = new opensearchserverless.CfnCollection(this, `opensearch-correction-for-${projectName}`, {
      name: collectionName,    
      description: `opensearch correction for ${projectName}`,
      standbyReplicas: 'DISABLED',
      type: 'VECTORSEARCH',
    });
    const collectionArn = OpenSearchCollection.attrArn

    new cdk.CfnOutput(this, `OpensearchCollectionEndpoint-${projectName}`, {
      value: OpenSearchCollection.attrCollectionEndpoint,
      description: 'The endpoint of opensearch correction',
    });

    const encPolicyName = `encription-${projectName}`
    const encPolicy = new opensearchserverless.CfnSecurityPolicy(this, `opensearch-encription-security-policy-for-${projectName}`, {
      name: encPolicyName,
      type: "encryption",
      description: `opensearch encryption policy for ${projectName}`,
      policy:
        '{"Rules":[{"ResourceType":"collection","Resource":["collection/*"]}],"AWSOwnedKey":true}',      
    });
    OpenSearchCollection.addDependency(encPolicy);

    const netPolicyName = `network-${projectName}`
    const netPolicy = new opensearchserverless.CfnSecurityPolicy(this, `opensearch-network-security-policy-for-${projectName}`, {
      name: netPolicyName,
      type: 'network',    
      description: `opensearch network policy for ${projectName}`,
      policy: JSON.stringify([
        {
          Rules: [
            {
              ResourceType: "dashboard",
              Resource: [`collection/${collectionName}`],
            },
            {
              ResourceType: "collection",
              Resource: [`collection/${collectionName}`],              
            }
          ],
          AllowFromPublic: true,          
        },
      ]), 
      
    });
    OpenSearchCollection.addDependency(netPolicy);

    const account = new iam.AccountPrincipal(this.account)
    const dataAccessPolicyName = `data-${projectName}`
    const dataAccessPolicy = new opensearchserverless.CfnAccessPolicy(this, `opensearch-data-collection-policy-for-${projectName}`, {
      name: dataAccessPolicyName,
      type: "data",
      policy: JSON.stringify([
        {
          Rules: [
            {
              Resource: [`collection/${collectionName}`],
              Permission: [
                "aoss:CreateCollectionItems",
                "aoss:DeleteCollectionItems",
                "aoss:UpdateCollectionItems",
                "aoss:DescribeCollectionItems",
              ],
              ResourceType: "collection",
            },
            {
              Resource: [`index/${collectionName}/*`],
              Permission: [
                "aoss:CreateIndex",
                "aoss:DeleteIndex",
                "aoss:UpdateIndex",
                "aoss:DescribeIndex",
                "aoss:ReadDocument",
                "aoss:WriteDocument",
              ], 
              ResourceType: "index",
            }
          ],
          Principal: [
            account.arn
          ], 
        },
      ]),
    });
    OpenSearchCollection.addDependency(dataAccessPolicy);

    // s3 
    const s3Bucket = new s3.Bucket(this, `storage-${projectName}`,{
      bucketName: bucketName,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      autoDeleteObjects: true,
      publicReadAccess: false,
      versioned: false,
      cors: [
        {
          allowedHeaders: ['*'],
          allowedMethods: [
            s3.HttpMethods.POST,
            s3.HttpMethods.PUT,
          ],
          allowedOrigins: ['*'],
        },
      ],
    });
    new cdk.CfnOutput(this, 'bucketName', {
      value: s3Bucket.bucketName,
      description: 'The nmae of bucket',
    });
    
    // EC2 Role
    const ec2Role = new iam.Role(this, `role-ec2-for-${projectName}`, {
      roleName: `role-ec2-for-${projectName}-${region}`,
      assumedBy: new iam.CompositePrincipal(
        new iam.ServicePrincipal("ec2.amazonaws.com"),
        new iam.ServicePrincipal("bedrock.amazonaws.com"),
      ),
      managedPolicies: [cdk.aws_iam.ManagedPolicy.fromAwsManagedPolicyName('CloudWatchAgentServerPolicy')] 
    });

    const secreatManagerPolicy = new iam.PolicyStatement({  
      resources: ['*'],
      actions: ['secretsmanager:GetSecretValue'],
    });       
    ec2Role.attachInlinePolicy( // for isengard
      new iam.Policy(this, `secret-manager-policy-ec2-for-${projectName}`, {
        statements: [secreatManagerPolicy],
      }),
    );  
    
    // Secret
    const weatherApiSecret = new secretsmanager.Secret(this, `weather-api-secret-for-${projectName}`, {
      description: 'secret for weather api key', // openweathermap
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      secretName: `openweathermap-${projectName}`,
      secretObjectValue: {
        project_name: cdk.SecretValue.unsafePlainText(projectName),
        weather_api_key: cdk.SecretValue.unsafePlainText(''),
      },
    });
    weatherApiSecret.grantRead(ec2Role) 

    const langsmithApiSecret = new secretsmanager.Secret(this, `weather-langsmith-secret-for-${projectName}`, {
      description: 'secret for lamgsmith api key', // openweathermap
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      secretName: `langsmithapikey-${projectName}`,
      secretObjectValue: {
        langchain_project: cdk.SecretValue.unsafePlainText(projectName),
        langsmith_api_key: cdk.SecretValue.unsafePlainText(''),
      }, 
    });
    langsmithApiSecret.grantRead(ec2Role) 

    const tavilyApiSecret = new secretsmanager.Secret(this, `weather-tavily-secret-for-${projectName}`, {
      description: 'secret for lamgsmith api key', // openweathermap
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      secretName: `tavilyapikey-${projectName}`,
      secretObjectValue: {
        project_name: cdk.SecretValue.unsafePlainText(projectName),
        tavily_api_key: cdk.SecretValue.unsafePlainText(''),
      },
    });
    tavilyApiSecret.grantRead(ec2Role) 

    const pvrePolicy = new iam.PolicyStatement({  
      resources: ['*'],
      actions: ['ssm:*', 'ssmmessages:*', 'ec2messages:*', 'tag:*'],
    });       
    ec2Role.attachInlinePolicy( // for isengard
      new iam.Policy(this, `pvre-policy-ec2-for-${projectName}`, {
        statements: [pvrePolicy],
      }),
    );  

    const BedrockPolicy = new iam.PolicyStatement({  
      resources: ['*'],
      actions: ['bedrock:*'],
    });        
    ec2Role.attachInlinePolicy( // add bedrock policy
      new iam.Policy(this, `bedrock-policy-ec2-for-${projectName}`, {
        statements: [BedrockPolicy],
      }),
    );     

    const ec2Policy = new iam.PolicyStatement({  
      resources: ['arn:aws:ec2:*:*:instance/*'],
      actions: ['ec2:*'],
    });
    ec2Role.attachInlinePolicy( // add bedrock policy
      new iam.Policy(this, `ec2-policy-for-${projectName}`, {
        statements: [ec2Policy],
      }),
    );

    // pass role
    const passRoleResourceArn = knowledge_base_role.roleArn;
    const passRolePolicy = new iam.PolicyStatement({  
      resources: [passRoleResourceArn],      
      actions: ['iam:PassRole'],
    });      
    ec2Role.attachInlinePolicy( // add pass role policy
      new iam.Policy(this, `pass-role-for-${projectName}`, {
      statements: [passRolePolicy],
      }), 
    );  
    const passRoleResourceArn = knowledge_base_role.roleArn;
    const passRolePolicy = new iam.PolicyStatement({  
      resources: [passRoleResourceArn],      
      actions: ['iam:PassRole'],
    });      
    ec2Role.attachInlinePolicy( // add pass role policy
      new iam.Policy(this, `pass-role-for-${projectName}`, {
      statements: [passRolePolicy],
      }), 
    );  

    // aoss
    const aossRolePolicy = new iam.PolicyStatement({  
      resources: ['*'],      
      actions: ['aoss:*'],
    }); 
    ec2Role.attachInlinePolicy( 
      new iam.Policy(this, `aoss-policy-for-${projectName}`, {
        statements: [aossRolePolicy],
      }),
    ); 

    // getRole
    const getRolePolicy = new iam.PolicyStatement({  
      resources: ['*'],      
      actions: ['iam:GetRole'],
    }); 
    ec2Role.attachInlinePolicy( 
      new iam.Policy(this, `getRole-policy-for-${projectName}`, {
        statements: [getRolePolicy],
      }),
    ); 

    // VPC
    const vpc = new ec2.Vpc(this, `vpc-for-${projectName}`, {
      vpcName: `vpc-for-${projectName}`,
      maxAzs: 2,
      ipAddresses: ec2.IpAddresses.cidr("20.64.0.0/16"),
      natGateways: 1,
      createInternetGateway: true,
      subnetConfiguration: [
        {
          cidrMask: 24,
          name: `public-subnet-for-${projectName}`,
          subnetType: ec2.SubnetType.PUBLIC
        }, 
        {
          cidrMask: 24,
          name: `private-subnet-for-${projectName}`,
          subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS
        }
      ]
    });  

    // S3 endpoint
    const s3BucketAcessPoint = vpc.addGatewayEndpoint(`s3Endpoint-${projectName}`, {
      service: ec2.GatewayVpcEndpointAwsService.S3,
    });

    s3BucketAcessPoint.addToPolicy(
      new iam.PolicyStatement({
        principals: [new iam.AnyPrincipal()],
        actions: ['s3:*'],
        resources: ['*'],
      }),
    ); 

    const bedrockEndpoint = vpc.addInterfaceEndpoint(`bedrock-endpoint-${projectName}`, {
      privateDnsEnabled: true,
      service: new ec2.InterfaceVpcEndpointService(`com.amazonaws.${region}.bedrock-runtime`, 443)
    });
    bedrockEndpoint.connections.allowDefaultPortFrom(ec2.Peer.ipv4(vpc.vpcCidrBlock), `allowBedrockPortFrom-${projectName}`)

    bedrockEndpoint.addToPolicy(
      new iam.PolicyStatement({
        principals: [new iam.AnyPrincipal()],
        actions: ['bedrock:*'],
        resources: ['*'],
      }),
    ); 

    // EC2 Security Group
    const ec2Sg = new ec2.SecurityGroup(this, `ec2-sg-for-${projectName}`,
      {
        vpc: vpc,
        allowAllOutbound: true,
        description: "Security group for ec2",
        securityGroupName: `ec2-sg-for-${projectName}`,
      }
    );
    // ec2Sg.addIngressRule(
    //   ec2.Peer.anyIpv4(),
    //   ec2.Port.tcp(22),
    //   'SSH',
    // );
    // ec2Sg.addIngressRule(
    //   ec2.Peer.anyIpv4(),
    //   ec2.Port.tcp(80),
    //   'HTTP',
    // );

    // ALB SG
    const albSg = new ec2.SecurityGroup(this, `alb-sg-for-${projectName}`, {
      vpc: vpc,
      allowAllOutbound: true,
      securityGroupName: `alb-sg-for-${projectName}`,
      description: 'security group for alb'
    });
    ec2Sg.connections.allowFrom(albSg, ec2.Port.tcp(targetPort), 'allow traffic from alb') // alb -> ec2

    // ALB
    const alb = new elbv2.ApplicationLoadBalancer(this, `alb-for-${projectName}`, {
      internetFacing: true,
      vpc: vpc,
      vpcSubnets: {
        subnets: vpc.publicSubnets
      },
      securityGroup: albSg,
      loadBalancerName: `alb-for-${projectName}`
    });
    alb.applyRemovalPolicy(cdk.RemovalPolicy.DESTROY); 

    new cdk.CfnOutput(this, `albUrl-for-${projectName}`, {
      value: `http://${alb.loadBalancerDnsName}/`,
      description: `albUrl-${projectName}`,
      exportName: `albUrl-${projectName}`
    }); 

    // CloudFront
    const CUSTOM_HEADER_NAME = "X-Custom-Header"
    const CUSTOM_HEADER_VALUE = `${projectName}_12dab15e4s31` // Temporary value
    const origin = new origins.LoadBalancerV2Origin(alb, {      
      httpPort: 80,
      customHeaders: {[CUSTOM_HEADER_NAME] : CUSTOM_HEADER_VALUE},
      originShieldEnabled: false,
      protocolPolicy: cloudFront.OriginProtocolPolicy.HTTP_ONLY      
    });
    const distribution = new cloudFront.Distribution(this, `cloudfront-for-${projectName}`, {
      comment: `Distribution-for-${projectName}`,
      defaultBehavior: {
        origin: origin,
        viewerProtocolPolicy: cloudFront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
        allowedMethods: cloudFront.AllowedMethods.ALLOW_ALL,
        cachePolicy: cloudFront.CachePolicy.CACHING_DISABLED,
        originRequestPolicy: cloudFront.OriginRequestPolicy.ALL_VIEWER        
      },
      priceClass: cloudFront.PriceClass.PRICE_CLASS_200
    }); 
    new cdk.CfnOutput(this, `distributionDomainName-for-${projectName}`, {
      value: 'https://'+distribution.domainName,
      description: 'The domain name of the Distribution'
    });
    
    // cloudfront for sharing s3
    const distribution_sharing = new cloudFront.Distribution(this, `sharing-for-${projectName}`, {
      defaultBehavior: {
        origin: origins.S3BucketOrigin.withOriginAccessControl(s3Bucket),
        allowedMethods: cloudFront.AllowedMethods.ALLOW_ALL,
        cachePolicy: cloudFront.CachePolicy.CACHING_DISABLED,
        viewerProtocolPolicy: cloudFront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
      },
      priceClass: cloudFront.PriceClass.PRICE_CLASS_200,  
    });
    new cdk.CfnOutput(this, `distribution-sharing-DomainName-for-${projectName}`, {
      value: 'https://'+distribution_sharing.domainName,
      description: 'The domain name of the Distribution Sharing',
    });

    // agent role
    const agent_role = new iam.Role(this,  `role-agent-for-${projectName}`, {
      roleName: `role-agent-for-${projectName}-${region}`,
      assumedBy: new iam.CompositePrincipal(
        new iam.ServicePrincipal("bedrock.amazonaws.com")
      )
    });

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

    const agentAliasPolicy = new iam.PolicyStatement({ 
      effect: iam.Effect.ALLOW,
      resources: [
        `arn:aws:bedrock:${region}:${accountId}:agent-alias/*`
      ],
      actions: [
        "bedrock:GetAgentAlias",
        "bedrock:InvokeAgent"
      ],
    });        
    agent_role.attachInlinePolicy( 
      new iam.Policy(this, `agent-alias-policy-for-${projectName}`, {
        statements: [agentAliasPolicy],
      }),
    );  

    const passRoleOfAgentPolicy = new iam.PolicyStatement({  
      resources: [`arn:aws:iam::${accountId}:role/role-agent-for-${projectName}-${region}`],      
      actions: ['iam:PassRole'],
    });      
    agent_role.attachInlinePolicy( // add pass role policy
      new iam.Policy(this, `pass-role-of-agent-for-${projectName}`, {
      statements: [passRoleOfAgentPolicy],
      }), 
    );  

    // lambda-tool
    const roleLambdaTools = new iam.Role(this, `role-lambda-tools-for-${projectName}`, {
      roleName: `role-lambda-tools-for-${projectName}-${region}`,
      assumedBy: new iam.CompositePrincipal(
        new iam.ServicePrincipal("lambda.amazonaws.com"),
        new iam.ServicePrincipal("bedrock.amazonaws.com"),
      ),
      // managedPolicies: [cdk.aws_iam.ManagedPolicy.fromAwsManagedPolicyName('CloudWatchLogsFullAccess')] 
    });
    // roleLambdaTools.addManagedPolicy({  // grant log permission
    //   managedPolicyArn: 'arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole',
    // });
    const CreateLogPolicy = new iam.PolicyStatement({  
      resources: [`arn:aws:logs:${region}:${accountId}:*`],
      actions: ['logs:CreateLogGroup'],
    });        
    roleLambdaTools.attachInlinePolicy( 
      new iam.Policy(this, `create-log-policy-lambda-tools-for-${projectName}`, {
        statements: [CreateLogPolicy],
      }),
    );
    const CreateLogStreamPolicy = new iam.PolicyStatement({  
      resources: [`arn:aws:logs:${region}:${accountId}:log-group:/aws/lambda/*`],
      actions: ["logs:CreateLogStream","logs:PutLogEvents"],
    });        
    roleLambdaTools.attachInlinePolicy( 
      new iam.Policy(this, `create-stream-log-policy-lambda-tools-for-${projectName}`, {
        statements: [CreateLogStreamPolicy],
      }),
    );      
    weatherApiSecret.grantRead(roleLambdaTools) 
    tavilyApiSecret.grantRead(roleLambdaTools) 

    // bedrock
    roleLambdaTools.attachInlinePolicy( 
      new iam.Policy(this, `tool-bedrock-invoke-policy-for-${projectName}`, {
        statements: [bedrockInvokePolicy],
      }),
    );  
    roleLambdaTools.attachInlinePolicy( 
      new iam.Policy(this, `tool-bedrock-agent-opensearch-policy-for-${projectName}`, {
        statements: [knowledgeBaseOpenSearchPolicy],
      }),
    );  
    roleLambdaTools.attachInlinePolicy( 
      new iam.Policy(this, `tool-bedrock-agent-bedrock-policy-for-${projectName}`, {
        statements: [knowledgeBaseBedrockPolicy],
      }),
    );  
        
    // lambda - tools
    // const lambdaTools = new lambda.Function(this, `lambda-tools-for-${projectName}`, {
    //   description: 'action group - tools',
    //   functionName: `lambda-tools-for-${projectName}`,
    //   handler: 'lambda_function.lambda_handler',
    //   runtime: lambda.Runtime.PYTHON_3_12,
    //   role: roleLambdaTools,
    //   code: lambda.Code.fromAsset(path.join(__dirname, '../../lambda-tools')),
    //   timeout: cdk.Duration.seconds(60),
    //   environment: {
    //     bedrock_region: String(region),
    //     projectName: projectName
    //   }
    // });
    
    const lambdaTools = new lambda.DockerImageFunction(this, `lambda-tools-for-${projectName}`, {
      description: 'action group - tools',
      functionName: `lambda-tools-for-${projectName}`,
      code: lambda.DockerImageCode.fromImageAsset(path.join(__dirname, '../../lambda-tools')),
      timeout: cdk.Duration.seconds(120),
      role: roleLambdaTools,
      environment: {
        bedrock_region: String(region),
        projectName: projectName,
        "sharing_url": 'https://'+distribution_sharing.domainName,
      }
    });     
    
    // lambdaTools.addPermission(`lambda-tools-permission-for-${projectName}`, {      
    //   principal: new iam.ServicePrincipal('bedrock.amazonaws.com'),
    //   action: 'lambda:InvokeFunction'
    // })
    lambdaTools.grantInvoke(new cdk.aws_iam.ServicePrincipal("bedrock.amazonaws.com")); 

    // user data for setting EC2
    const userData = ec2.UserData.forLinux();

    const environment = {
      "projectName": projectName,
      "accountId": accountId,
      "region": region,
      "knowledge_base_role": knowledge_base_role.roleArn,
      "collectionArn": collectionArn,
      "agent_role_arn": agent_role.roleArn,
      "opensearch_url": OpenSearchCollection.attrCollectionEndpoint,
      "s3_bucket": s3Bucket.bucketName,      
      "s3_arn": s3Bucket.bucketArn,
      "sharing_url": 'https://'+distribution_sharing.domainName,
      "lambda-tools": lambdaTools.functionArn
    }    
    new cdk.CfnOutput(this, `environment-for-${projectName}`, {
      value: JSON.stringify(environment),
      description: `environment-${projectName}`,
      exportName: `environment-${projectName}`
    });

    const commands = [
      // 'yum install nginx -y',
      // 'service nginx start',
      'yum install git python-pip -y',
      'pip install pip --upgrade',            
      `sh -c "cat <<EOF > /etc/systemd/system/streamlit.service
[Unit]
Description=Streamlit
After=network-online.target

[Service]
User=ec2-user
Group=ec2-user
Restart=always
ExecStart=/home/ec2-user/.local/bin/streamlit run /home/ec2-user/${projectName}/application/app.py

[Install]
WantedBy=multi-user.target
EOF"`,
        `runuser -l ec2-user -c "mkdir -p /home/ec2-user/.streamlit"`,
        `runuser -l ec2-user -c 'cat <<EOF > /home/ec2-user/.streamlit/config.toml
[server]
port=${targetPort}
maxUploadSize = 50

[theme]
base="dark"
primaryColor="#fff700"
EOF'`,
      `json='${JSON.stringify(environment)}' && echo "$json">/home/config.json`,
      `runuser -l ec2-user -c 'cd && git clone https://github.com/kyopark2014/${projectName}'`,
      `runuser -l ec2-user -c 'pip install streamlit streamlit_chat beautifulsoup4 pytz tavily-python'`,        
      `runuser -l ec2-user -c 'pip install boto3 langchain_aws langchain langchain_community langgraph opensearch-py PyPDF2'`,
      'systemctl enable streamlit.service',
      'systemctl start streamlit',
      `yum install -y amazon-cloudwatch-agent`,
      `mkdir /var/log/application/ && chown ec2-user /var/log/application && chgrp ec2-user /var/log/application`,
    ];    
    userData.addCommands(...commands);

    // EC2 instance
    const appInstance = new ec2.Instance(this, `app-for-${projectName}`, {
      instanceName: `app-for-${projectName}`,
      instanceType: new ec2.InstanceType('t2.small'), // m5.large
      // instanceType: ec2.InstanceType.of(ec2.InstanceClass.T2, ec2.InstanceSize.SMALL),
      machineImage: new ec2.AmazonLinuxImage({
        generation: ec2.AmazonLinuxGeneration.AMAZON_LINUX_2023
      }),
      // machineImage: ec2.MachineImage.latestAmazonLinux2023(),
      vpc: vpc,
      vpcSubnets: {
        subnets: vpc.privateSubnets  
      },
      securityGroup: ec2Sg,
      role: ec2Role,
      userData: userData,
      blockDevices: [{
        deviceName: '/dev/xvda',
        volume: ec2.BlockDeviceVolume.ebs(8, {
          deleteOnTermination: true,
          encrypted: true,
        }),
      }],
      detailedMonitoring: true,
      instanceInitiatedShutdownBehavior: ec2.InstanceInitiatedShutdownBehavior.TERMINATE,
    }); 
    s3Bucket.grantReadWrite(appInstance);
    appInstance.applyRemovalPolicy(cdk.RemovalPolicy.DESTROY);

    // ALB Target
    const targets: elbv2_tg.InstanceTarget[] = new Array();
    targets.push(new elbv2_tg.InstanceTarget(appInstance)); 

    // ALB Listener
    const listener = alb.addListener(`HttpListener-for-${projectName}`, {   
      port: 80,
      open: true
    });     
    const targetGroup = listener.addTargets(`WebEc2Target-for-${projectName}`, {
      targetGroupName: `TG-for-${projectName}`,
      targets: targets,
      protocol: elbv2.ApplicationProtocol.HTTP,
      port: targetPort,
      conditions: [elbv2.ListenerCondition.httpHeader(CUSTOM_HEADER_NAME, [CUSTOM_HEADER_VALUE])],
      priority: 10      
    });
    listener.addTargetGroups("demoTargetGroupInt", {
      targetGroups: [targetGroup]
    })
    const defaultAction = elbv2.ListenerAction.fixedResponse(403, {
        contentType: "text/plain",
        messageBody: 'Access denied',
    })
    listener.addAction(`RedirectHttpListener-for-${projectName}`, {
      action: defaultAction
    });       
  }
}