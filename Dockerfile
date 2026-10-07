FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Non-root user; runtime/, data/ and dashboard/data/ are the only writable trees.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/runtime /app/data /app/dashboard/data \
    && chown -R app:app /app/runtime /app/data /app/dashboard/data
USER app

# The same image runs both processes; compose picks the command.
#   orchestrator + UI : python -m roa        (8100)
#   dashboard         : python -m dashboard  (8200)
EXPOSE 8100 8200
CMD ["python", "-m", "roa"]
