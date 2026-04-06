FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip && pip install .

EXPOSE 7860
ENV PORT=7860
CMD ["sh", "-c", "uvicorn supportdesk_env.server.app:app --host 0.0.0.0 --port ${PORT:-7860}"]
