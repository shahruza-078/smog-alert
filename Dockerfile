FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# the database lives on the Fly volume mounted at /data (survives restarts and deploys)
ENV DATABASE_PATH=/data/smog.db PORT=8080
EXPOSE 8080
CMD ["python", "app.py"]
