FROM python:3.11-slim

WORKDIR /app

COPY requirements-api.txt .
COPY pyproject.toml .

RUN pip install --no-cache-dir -r requirements-api.txt

COPY app/api ./app/api
COPY app/rag ./app/rag
COPY app/core ./app/core

RUN pip install --no-cache-dir .

CMD ["uvicorn", "app.api.api:app", "--host", "0.0.0.0", "--port", "8000"]