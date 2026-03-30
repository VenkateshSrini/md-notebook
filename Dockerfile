FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY vectorizer/ vectorizer/
COPY notebook_lm/ notebook_lm/
COPY notebook_api/ notebook_api/
COPY notebook_ui/ notebook_ui/
COPY vector-db/ vector-db/
COPY main_api.py .
COPY main_ui.py .

# REST API port (FastAPI/uvicorn)
ENV PORT=8000
# Gradio UI port
ENV UI_HOST=0.0.0.0
ENV UI_PORT=7860

EXPOSE ${PORT}
EXPOSE ${UI_PORT}

# Default: run the REST API server.
# To run the Gradio UI instead, override CMD at container run time:
#   docker run -e ANTHROPIC_API_KEY=... -p 7860:7860 <image> python main_ui.py
CMD ["python", "main_api.py"]
