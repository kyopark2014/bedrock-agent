FROM amazon/aws-lambda-python:3.12

WORKDIR /var/task/lambda-tools

COPY lambda_function.py /var/task/
COPY . ..

RUN /var/lang/bin/python3 -m pip install beautifulsoup4
RUN /var/lang/bin/python3 -m pip install request

CMD ["lambda_function.lambda_handler"]