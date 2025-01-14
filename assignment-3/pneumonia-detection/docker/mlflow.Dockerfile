FROM python:3.11-slim

WORKDIR /app

RUN pip install mlflow psutil

EXPOSE 5000

CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--backend-store-uri", "${BACKEND_STORE_URI}", \
     "--default-artifact-root", "${DEFAULT_ARTIFACT_ROOT}"]