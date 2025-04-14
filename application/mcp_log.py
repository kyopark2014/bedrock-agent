
import json
import boto3
import logging
import sys
from typing import Dict, Optional, Any
from datetime import datetime, timedelta

logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("mcp-log")

async def list_groups(
    prefix: Optional[str] = None,
    region: Optional[str] = 'us-west-2'
) -> str:
    """
    List available CloudWatch log groups.
    Parameters:
        prefix: the prefix of bucket  
        region: The region of aws infrastructure, e.g. us-west-2
    """

    logger.info(f"list_groups: prefix={prefix}, region={region}")

    log_client = boto3.client(
        service_name='logs',
        region_name=region
    )

    kwargs = {}
    if prefix:
        kwargs["logGroupNamePrefix"] = prefix

    response = log_client.describe_log_groups(**kwargs)
    log_groups = response.get("logGroups", [])

    # Format the response
    # formatted_groups = []
    # for group in log_groups:
    #     formatted_groups.append(
    #         {
    #             "logGroupName": group.get("logGroupName"),
    #             "creationTime": group.get("creationTime"),
    #             "storedBytes": group.get("storedBytes"),
    #         }
    #     )

    # response_json = json.dumps(formatted_groups, ensure_ascii=True)
    # logger.info(f"response: {response_json}")

    # return response_json

    result = ""
    for group in log_groups:
        result += f"logGroupName: {group.get('logGroupName')}\n"
        result += f"creationTime: {group.get('creationTime')}\n"
        result += f"storedBytes: {group.get('storedBytes')}\n\n"

    logger.info(f"response: {result}")

    return result

def _parse_relative_time(time_str: str) -> Optional[int]:
    """Parse a relative time string into a timestamp."""
    if not time_str:
        return None

    logger.debug(f"Parsing time string: {time_str}")

    # Check if it's an ISO format date
    try:
        dt = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
        timestamp = int(dt.timestamp() * 1000)
        logger.debug(f"Parsed ISO format date: {dt.isoformat()}, timestamp: {timestamp}")
        return timestamp
    except ValueError:
        logger.debug(f"Not an ISO format date, trying relative time format")
        pass

    # Parse relative time
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    if time_str[-1] in units and time_str[:-1].isdigit():
        value = int(time_str[:-1])
        unit = time_str[-1]
        seconds = value * units[unit]
        dt = datetime.now() - timedelta(seconds=seconds)
        timestamp = int(dt.timestamp() * 1000)
        logger.debug(f"Parsed relative time: {value}{unit}, timestamp: {timestamp}")
        return timestamp

    error_msg = f"Invalid time format: {time_str}"
    logger.error(error_msg)
    raise ValueError(error_msg)

async def get_logs(
    logGroupName: str,
    logStreamName: Optional[str] = None,
    startTime: Optional[str] = None,
    endTime: Optional[str] = None,
    region: Optional[str] = 'us-west-2'
) -> str:
    """
    Get CloudWatch logs from a specific log group and stream.
    Parameters:
        logGroupName: the name of log group
        logStreamName: the stream name of log
        startTime: start time of the log
        endTime: end time of the log
        region: the region name of aws
    """
    logger.info(
        f"Getting CloudWatch logs for group: {logGroupName}, stream: {logStreamName}, "
        f"startTime: {startTime}, endTime: {endTime}"
        f"region: {region}"
    )

    log_client = boto3.client(
        service_name='logs',
        region_name=region
    )

    # Parse start and end times
    start_time_ms = None
    if startTime:
        start_time_ms = _parse_relative_time(startTime)

    end_time_ms = None
    if endTime:
        end_time_ms = _parse_relative_time(endTime)

    # Get logs
    kwargs = {
        "logGroupName": logGroupName,
    }

    if logStreamName:
        kwargs["logStreamNames"] = [logStreamName]

    if start_time_ms:
        kwargs["startTime"] = start_time_ms

    if end_time_ms:
        kwargs["endTime"] = end_time_ms

    # Use filter_log_events for more flexible querying
    response = log_client.filter_log_events(**kwargs)
    events = response.get("events", [])

    # Format the response
    # formatted_events = []
    # for event in events:
    #     timestamp = event.get("timestamp")
    #     if timestamp:
    #         try:
    #             timestamp = datetime.fromtimestamp(timestamp / 1000).isoformat()
    #         except Exception:
    #             timestamp = str(timestamp)

    #     formatted_events.append(
    #         {
    #             "timestamp": timestamp,
    #             "message": event.get("message"),
    #             "logStreamName": event.get("logStreamName"),
    #         }
    #     )    

    # response_json = json.dumps(formatted_events, ensure_ascii=True, default=str)
    # logger.info(f"response: {response_json}")
    # return response_json

    result = ""
    for event in events:
        timestamp = event.get("timestamp")
        if timestamp:
            try:
                timestamp = datetime.fromtimestamp(timestamp / 1000).isoformat()
            except Exception:
                timestamp = str(timestamp)

        result += f"timestamp: {timestamp}\n"
        result += f"message: {event.get('message')}\n"
        result += f"logStreamName: {event.get('logStreamName')}\n\n"

    return result