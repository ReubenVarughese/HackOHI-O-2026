FROM python:3.11-slim

WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt
COPY api ./api
COPY model/restoration-extended-classifier.joblib ./model/restoration-extended-classifier.joblib
COPY model/restoration-positive-duration-xgboost.joblib ./model/restoration-positive-duration-xgboost.joblib
COPY model/historical-lookup.json ./model/historical-lookup.json

ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn api.prediction_api:app --host 0.0.0.0 --port ${PORT}"]
