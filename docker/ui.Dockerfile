FROM python:3.11-slim

WORKDIR /app

COPY requirements-ui.txt .
COPY pyproject.toml .

RUN pip install --no-cache-dir -r requirements-ui.txt

COPY app/ginterface ./app/ginterface
COPY app/core ./app/core

RUN pip install --no-cache-dir -e .

CMD ["streamlit", "run", "app/ginterface/gui.py", "--server.address=0.0.0.0", "--server.port=8501"]