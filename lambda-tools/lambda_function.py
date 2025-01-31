
import json
import requests
import traceback
from bs4 import BeautifulSoup

def lambda_handler(event, context):
    print('event: ', event)
    
    agent = event['agent']
    actionGroup = event['actionGroup']
    
    function = event['function']
    print('function: ', function)

    parameters = event.get('parameters', [])
    print('parameters: ', parameters)

    for parameter in parameters:
        print('parameter: ', parameter)
        name = parameter['name']
        value = parameter['value']

        if name == 'keyword':
            keyword = value
            keyword = keyword.replace('\'','')
            print(f"keyword: {keyword}")

            answer = ""
            url = f"https://search.kyobobook.co.kr/search?keyword={keyword}&gbCode=TOT&target=total"

            try:
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
            except Exception:
                err_msg = traceback.format_exc()
                print('error message: ', err_msg)      

    print("answer: ", answer)

    responseBody =  {
        "TEXT": {
            "body": answer
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
    print(f"response: {response}")
    
    return response
