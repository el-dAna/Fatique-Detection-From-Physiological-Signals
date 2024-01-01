FROM public.ecr.aws/lambda/python:3.8

RUN mkdir -p /app
COPY ./App.py /app/
COPY mylib/ /app/mylib/
COPY pages/ /app/pages/
COPY utils/ /app/utils/
COPY ./requirements.txt /app/requirements.txt

RUN pip install --upgrade -r /app/requirements.txt
WORKDIR /app

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "App.py", "--server.port=8501", "--server.address=0.0.0.0"]
