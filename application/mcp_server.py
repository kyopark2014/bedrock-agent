import logging
import sys
import json
import datetime
import mcp_log as log
import mcp_cost as cost
import mcp_rag as rag
import mcp_s3 as storage

from mcp import StdioServerParameters
from InlineAgent.tools import MCPStdio
from InlineAgent.action_group import ActionGroup
from InlineAgent.agent import InlineAgent
# from InlineAgent.knowledge_base import KnowledgeBase

logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("playwright")

def load_mcp_server_parameters(mcp_config):
    mcp_json = json.loads(mcp_config)
    logger.info(f"mcp_json: {mcp_json}")

    mcpServers = mcp_json.get("mcpServers")
    logger.info(f"mcpServers: {mcpServers}")

    command = ""
    args = []
    env = ""
    if mcpServers is not None:
        for server in mcpServers:
            logger.info(f"server: {server}")

            config = mcpServers.get(server)
            logger.info(f"config: {config}")

            if "command" in config:
                command = config["command"]
            if "args" in config:
                args = config["args"]
            if "env" in config:
                env = config["env"]
            break

    if env:
        return StdioServerParameters(
            command=command,
            args=args,
            env=env
        )
    else:
        return StdioServerParameters(
            command=command,
            args=args
        )

async def load_mcp_clients(mcp_config):
    logger.info(f"mcp_config: {mcp_config}")

    mcp_json = json.loads(mcp_config)
    logger.info(f"mcp_json: {mcp_json}")

    mcpServers = mcp_json.get("mcpServers")
    logger.info(f"mcpServers: {mcpServers}")
  
    mcp_clients = []
    for server in mcpServers:
        logger.info(f"server: {server}")

        config = mcpServers.get(server)
        logger.info(f"config: {config}")

        if "command" in config:
            command = config["command"]
        if "args" in config:
            args = config["args"]
        if "env" in config:
            env = config["env"]
            
            server_params = StdioServerParameters(
                command=command,
                args=args,
                env=env
            )
        else:
            server_params = StdioServerParameters(
                command=command,
                args=args
            )
        logger.info(f"server_params: {server_params}")
        tool_client = await MCPStdio.create(server_params=server_params)
        mcp_clients.append(tool_client)

    return mcp_clients

def get_time() -> dict:
    """Get the current time"""
    now = datetime.datetime.now()

    logger.info(f"Current time: {now}")

    #return {"time": now.strftime("%H:%M:%S")}
    return now.strftime("%H:%M:%S")

# restaurant_kb = KnowledgeBase(
#     name="restaurant-kb",
#     description="Use this knowledgebase to get information about restaurants menu.",
#     additional_props={
#         "retrievalConfiguration": {"vectorSearchConfiguration": {"numberOfResults": 5}}
#     },
# )

import uuid
session_id = f"session-{str(uuid.uuid4())}"
async def run(text, mcp_config, st):
    # server_params = load_mcp_server_parameters(mcp_config) # single
    # tool_client = await MCPStdio.create(server_params=server_params)

    mcp_clients = await load_mcp_clients(mcp_config) # multiple

    try:
        # Define an action group
        tool_action_group = ActionGroup(
            name="ToolActionGroup",
            description="retrieve information using tools",
            mcp_clients=mcp_clients,
        )

        time_group = ActionGroup(
            name="TimeService",
            description="Provides time-related information",
            tools=[get_time],
        )

        cost_group = ActionGroup(
            name="costService",
            description="tools for AWS",
            tools=[cost.get_cost_analysis],
        )

        list_log_group = ActionGroup(
            name="ListLogService",
            description="earn the list of log groups on AWS",
            tools=[log.list_groups],
        )

        get_log_group = ActionGroup(
            name="GetLogService",
            description="get the log of the group on AWS",
            tools=[log.get_logs],
        )

        list_bucket_group = ActionGroup(
            name="ListBucketService",
            description="retrieve the bucket information on AWS such as bucket name, object and resources.",
            tools=[storage.list_buckets, storage.list_objects, storage.list_resources],
        )
        
        # Invoke agent
        result = await InlineAgent(
            foundation_model="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            instruction="""You are a friendly assistant that is responsible for resolving user queries. """,
            agent_name="tool_agent",
            #action_groups=[tool_action_group, time_group, aws_group],
            action_groups=[
                tool_action_group, 
                time_group, 
                cost_group, 
                list_log_group, 
                get_log_group, 
                list_bucket_group
            ],
        ).invoke(
            input_text=text,
            enable_trace = True,
            add_citation=True,
            session_id=session_id
        )

        logger.info(f"result: {result}")
        st.markdown(result)

        image_url = []
        st.session_state.messages.append({
            "role": "assistant", 
            "content": result,
            "images": image_url if image_url else []
        })

    finally:
        for mcp_client in mcp_clients:
            await mcp_client.cleanup()
                
    return result, ""
