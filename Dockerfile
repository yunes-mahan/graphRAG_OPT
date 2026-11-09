# Reproducibility container:  docker build -t gcm .  &&  docker run --rm gcm
FROM python:3.12-slim
WORKDIR /work
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN chmod +x quick_reproduce.sh
CMD ["./quick_reproduce.sh"]
