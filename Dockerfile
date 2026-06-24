FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
COPY pyproject.toml .

RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY tests ./tests

CMD ["sh", "-c", "python -m ruff check . && python -m pytest"]