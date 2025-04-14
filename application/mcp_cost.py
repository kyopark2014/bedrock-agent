import json
import boto3
import logging
import sys
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
import pandas as pd

logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("mcp-cost")

def get_cost_analysis(days: int=30, region: str="us-west-2"):
    """
    Cost analysis data collection
    Parameters:
        days: the period of the data, e.g., 30
        region: The region of aws infrastructure, e.g., us-west-2
    """    
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # cost explorer
        ce = boto3.client(
            service_name='ce',
            region_name=region
        )

        # service cost
        service_response = ce.get_cost_and_usage(
            TimePeriod={
                'Start': start_date.strftime('%Y-%m-%d'),
                'End': end_date.strftime('%Y-%m-%d')
            },
            Granularity='MONTHLY',
            Metrics=['UnblendedCost'],
            GroupBy=[{'Type': 'DIMENSION', 'Key': 'SERVICE'}]
        )
        
        service_costs = pd.DataFrame([
            {
                'SERVICE': group['Keys'][0],
                'cost': float(group['Metrics']['UnblendedCost']['Amount'])
            }
            for group in service_response['ResultsByTime'][0]['Groups']
        ])
        logger.info(f"Service Cost: {service_response}")
        
        # region cost
        region_response = ce.get_cost_and_usage(
            TimePeriod={
                'Start': start_date.strftime('%Y-%m-%d'),
                'End': end_date.strftime('%Y-%m-%d')
            },
            Granularity='MONTHLY',
            Metrics=['UnblendedCost'],
            GroupBy=[{'Type': 'DIMENSION', 'Key': 'REGION'}]
        )
        logger.info(f"Region Cost: {region_response}")
        
        region_costs = pd.DataFrame([
            {
                'REGION': group['Keys'][0],
                'cost': float(group['Metrics']['UnblendedCost']['Amount'])
            }
            for group in region_response['ResultsByTime'][0]['Groups']
        ])
        
        # Daily Cost
        daily_response = ce.get_cost_and_usage(
            TimePeriod={
                'Start': start_date.strftime('%Y-%m-%d'),
                'End': end_date.strftime('%Y-%m-%d')
            },
            Granularity='DAILY',
            Metrics=['UnblendedCost'],
            GroupBy=[{'Type': 'DIMENSION', 'Key': 'SERVICE'}]
        )
        logger.info(f"Daily Cost: {daily_response}")
        
        daily_costs = []
        for time_period in daily_response['ResultsByTime']:
            date = time_period['TimePeriod']['Start']
            for group in time_period['Groups']:
                daily_costs.append({
                    'date': date,
                    'SERVICE': group['Keys'][0],
                    'cost': float(group['Metrics']['UnblendedCost']['Amount'])
                })
        
        daily_costs_df = pd.DataFrame(daily_costs)
        logger.info(f"Daily Cost (df): {daily_costs_df}")
        
        # return json.dumps({
        #     'service_costs': service_costs,
        #     'region_costs': region_costs,
        #     'daily_costs': daily_costs_df
        # })
        result = f"service_costs: {service_costs}\n\nregion_costs: {region_costs}\n\ndaily_costs: {daily_costs}"
        logger.info(f"result: {result}")
        return result
        
    except Exception as e:
        print(f"Error in cost analysis: {str(e)}")
        return None
