import logging
import sys

from typing import List, Optional, Any
import asyncio
import os
from mcp.types import Resource
import aioboto3

logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("mcp-s3")

session = aioboto3.Session()

def _get_configured_buckets() -> List[str]:
    """
    Get configured bucket names from environment variables.
    Format in .env file:
    S3_BUCKETS=bucket1,bucket2,bucket3
    or
    S3_BUCKET_1=bucket1
    S3_BUCKET_2=bucket2
    see env.example ############
    """
    # Try comma-separated list first
    bucket_list = os.getenv('S3_BUCKETS')
    if bucket_list:
        return [b.strip() for b in bucket_list.split(',')]

    buckets = []
    i = 1
    while True:
        bucket = os.getenv(f'S3_BUCKET_{i}')
        if not bucket:
            break
        buckets.append(bucket.strip())
        i += 1

    return buckets            

configured_buckets = _get_configured_buckets()

def is_text_file(key: str) -> bool:
    """Determine if a file is text-based by its extension"""
    text_extensions = {
        '.txt', '.log', '.json', '.xml', '.yml', '.yaml', '.md',
        '.csv', '.ini', '.conf', '.py', '.js', '.html', '.css',
        '.sh', '.bash', '.cfg', '.properties'
    }
    return any(key.lower().endswith(ext) for ext in text_extensions)

async def get_list_buckets(
    start_after: Optional[str] = None,
    max_buckets: Optional[int] = 10,
    region: Optional[str] = "us-west-2"
) -> List[dict]:
    """
    List S3 buckets using async client with pagination
    """
    async with session.client('s3', region_name=region) as s3:
        if configured_buckets:
            # If buckets are configured, only return those
            response = await s3.list_buckets()
            all_buckets = response.get('Buckets', [])
            configured_bucket_list = [
                bucket for bucket in all_buckets
                if bucket['Name'] in configured_buckets
            ]

            if start_after:
                configured_bucket_list = [
                    b for b in configured_bucket_list
                    if b['Name'] > start_after
                ]

            return configured_bucket_list[:max_buckets]
        else:
            # Default behavior if no buckets configured
            response = await s3.list_buckets()
            buckets = response.get('Buckets', [])

            if start_after:
                buckets = [b for b in buckets if b['Name'] > start_after]

            return buckets[:max_buckets]

async def list_buckets(
    start_after: Optional[str] = None,
    max_buckets: Optional[int] = 10,
    region: Optional[str] = "us-west-2"
) -> str:
    """
    List S3 buckets using async client with pagination
    Parameters:
        start_after: Start listing after this bucket name
        max_buckets: the number of buckets 
        region: the region of aws infrastructure, e.g. us-west-2
    Returns:
        List of bucket dictionaries containing Name and CreationDate
    """

    buckets = await get_list_buckets(start_after, max_buckets, region)
    
    # 로깅을 위한 문자열 생성
    result = ""
    for bucket in buckets:
        result += f"Bucket Name: {bucket['Name']}, Creation Date: {bucket['CreationDate']}\n"
    logger.info(f"Buckets found: {result}")
    
    return result

async def get_list_objects(
    bucket_name: str, 
    prefix: Optional[str] = "", 
    max_keys: Optional[int] = 1000,
    region: Optional[str] = "us-west-2"
) -> List[dict]:
    """
    List objects in a specific bucket using async client with pagination
    Args:
        bucket_name: Name of the S3 bucket
        prefix: Object prefix for filtering
        max_keys: Maximum number of keys to return,
        region: Name of the aws region
    """
    if configured_buckets and bucket_name not in configured_buckets:
        logger.warning(f"Bucket {bucket_name} not in configured bucket list")
        return []

    async with session.client('s3', region_name=region) as s3:
        response = await s3.list_objects_v2(
            Bucket=bucket_name,
            Prefix=prefix,
            MaxKeys=max_keys
        )
        return response.get('Contents', [])

async def list_objects(
    bucket_name: str, 
    prefix: Optional[str] = "", 
    max_keys: Optional[int] = 1000,
    region: Optional[str] = "us-west-2"
) -> str:
    """
    List objects in a specific bucket using async client with pagination
    Parameters:
        bucket_name: Name of the S3 bucket
        prefix: Object prefix for filtering
        max_keys: Maximum number of keys to return,
        region: the name of aws region
    """
    
    response = await get_list_objects(bucket_name, prefix, max_keys, region)

    result = ""
    for obj in response:
        result += f"Key: {obj['Key']}, Size: {obj['Size']}, LastModified: {obj['LastModified']}\n"
    
    return result

async def get_list_resources(
    start_after: Optional[str] = None,
    max_buckets: Optional[int] = 10,
    region: Optional[str] = "us-west-2"
) -> List[Resource]:
    """
    List S3 buckets and their contents as resources with pagination
    Args:
        start_after: Start listing after this bucket name
    """
    resources = []
    logger.debug("Starting to list resources")
    
    logger.debug(f"Configured buckets: {configured_buckets}")

    try:
        # Get limited number of buckets
        buckets = await get_list_buckets(start_after, max_buckets, region)
        logger.debug(f"Processing {len(buckets)} buckets (max: {max_buckets})")

        # limit concurrent operations
        async def process_bucket(bucket):
            bucket_name = bucket['Name']
            logger.debug(f"Processing bucket: {bucket_name}")

            try:
                # List objects in the bucket with a reasonable limit
                objects = await get_list_objects(bucket_name, max_keys=1000)

                for obj in objects:
                    if 'Key' in obj and not obj['Key'].endswith('/'):
                        object_key = obj['Key']
                        mime_type = "text/plain" if is_text_file(object_key) else "text/markdown"

                        resource = Resource(
                            uri=f"s3://{bucket_name}/{object_key}",
                            name=object_key,
                            mimeType=mime_type
                        )
                        resources.append(resource)
                        logger.debug(f"Added resource: {resource.uri}")

            except Exception as e:
                logger.error(f"Error listing objects in bucket {bucket_name}: {str(e)}")

        # Use semaphore to limit concurrent bucket processing
        semaphore = asyncio.Semaphore(3)  # Limit concurrent bucket processing
        async def process_bucket_with_semaphore(bucket):
            async with semaphore:
                await process_bucket(bucket)

        # Process buckets concurrently
        await asyncio.gather(*[process_bucket_with_semaphore(bucket) for bucket in buckets])

    except Exception as e:
        logger.error(f"Error listing buckets: {str(e)}")
        raise

    logger.info(f"Returning {len(resources)} resources")
    return resources

def list_resources(    
    start_after: Optional[str] = None,
    max_buckets: Optional[int] = 10,
    region: Optional[str] = "us-west-2"
) -> str:
    """
    List S3 buckets and their contents as resources with pagination
    Parameters:
        start_after: Start listing after this bucket name
        max_buckets: the number of buckets 
        region: the region of aws infrastructure, e.g. us-west-2
    """    
    # 비동기 함수를 동기적으로 실행
    loop = asyncio.get_event_loop()
    resources = loop.run_until_complete(get_list_resources(start_after, max_buckets, region))
    
    result = ""
    for resource in resources:
        result += f"name: {resource.name}, mimeType: {resource.mimeType}, uri: {resource.uri}\n"
    
    return result
