FROM public.ecr.aws/lambda/python:3.8

RUN mkdir -p /App
COPY ./App.py /App/
COPY mylib/ /App/mylib/
COPY pages/ /App/pages/
COPY utils/ /App/utils/
COPY ./requirements.txt /App/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /App/requirements.txt
WORKDIR /App

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "App.py", "--server.port=8501", "--server.address=0.0.0.0"]
