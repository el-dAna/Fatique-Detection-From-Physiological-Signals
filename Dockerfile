FROM public.ecr.aws/lambda/python:3.8

RUN mkdir -p /app
COPY ./app.py /app/
COPY mylib/ /app/mylib/
COPY pages/ /app/pages/
COPY utils/ /app/utils/
COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt
WORKDIR /app
EXPOSE 8080
CMD [ "app.py" ]
ENTRYPOINT [ "python" ]