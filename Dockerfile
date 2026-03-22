FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY vectorizer/ vectorizer/
COPY notebook_lm/ notebook_lm/
COPY notebook_api/ notebook_api/
COPY vector-db/ vector-db/
COPY main_api.py .

EXPOSE 8000

CMD ["python", "main_api.py"]
