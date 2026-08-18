FROM python:3.11-slim

WORKDIR /app

COPY requirements-api.txt .

RUN pip install --default-timeout=1000 --no-cache-dir -r requirements-api.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "deployment.fastapi.app:app", "--host", "0.0.0.0", "--port", "8000"]
