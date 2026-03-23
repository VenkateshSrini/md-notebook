FROM python:3.14-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY vectorizer/ vectorizer/
COPY notebook_lm/ notebook_lm/
COPY notebook_api/ notebook_api/
COPY vector-db/ vector-db/
COPY main_api.py .

ENV PORT=8000

EXPOSE ${PORT}

CMD ["python", "main_api.py"]
