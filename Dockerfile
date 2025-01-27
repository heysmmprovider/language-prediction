FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3.9-distutils \
    build-essential \
    nano \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install pip
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.9 get-pip.py && \
    rm get-pip.py

# Set Python aliases
RUN ln -sf /usr/bin/python3.9 /usr/bin/python
RUN ln -sf /usr/bin/pip3 /usr/bin/pip

# Install PyTorch with CUDA support first
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Download model during build (prevents download at runtime)
RUN python -c "from transformers import AutoModelForSequenceClassification, AutoTokenizer; model_name='papluca/xlm-roberta-base-language-detection'; AutoModelForSequenceClassification.from_pretrained(model_name); AutoTokenizer.from_pretrained(model_name)"

EXPOSE 10003

CMD ["python", "app.py"]