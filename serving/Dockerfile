FROM python:3.12.7-slim
WORKDIR /app
COPY serving/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY serving/credit_default_model.joblib /app/model/credit_default_model.joblib
COPY serving/7.Inference.py /app/7.Inference.py
ENV MODEL_PATH=/app/model/credit_default_model.joblib HOST=0.0.0.0 PORT=8000
EXPOSE 8000
CMD ["python", "/app/7.Inference.py"]