# app/Dockerfile

FROM python:3.9-slim

EXPOSE 8501

WORKDIR /app

COPY . .

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common

RUN pip3 install -r requirements.txt
# Make RUN commands use the new environment:
#ENV PATH='${PATH}:/app/.local/bin'

# Activate the environment, and make sure it's activated:
# Demonstrate the environment is activated:
#HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
# The code to run when container is started:
#ENTRYPOINT ["streamlit", "run", "st_app_updated.py", "--server.port=$PORT"]
CMD streamlit run --server.port $PORT st_app_updated.py