FROM amazon/aws-lambda-python:3.11

WORKDIR /var/task/lambda-tools

COPY lambda_function.py /var/task/
COPY . ..

RUN /var/lang/bin/python3 -m pip install beautifulsoup4

CMD ["lambda_function.lambda_handler"]