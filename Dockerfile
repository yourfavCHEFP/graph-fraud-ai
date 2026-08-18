FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git git-lfs

RUN git lfs install

COPY . .

RUN pip install --default-timeout=1000 --no-cache-dir -r requirements-api.txt

CMD ["uvicorn","deployment.fastapi.app:app","--host","0.0.0.0","--port","8000"]
