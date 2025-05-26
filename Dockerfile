FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    nmap libgl1 libglib2.0-0 --fix-missing \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install pyuwsgi -r requirements.txt


# Pre-download ResNet50 into the torch.hub cache
# so that at runtime we never do an HTTP fetch
RUN python3 - <<EOF
import torch
# This will pull both the repo and the weights into /root/.cache/torch/hub
torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
EOF

# Copy app files
COPY . .

# Set Flask environment
EXPOSE 8080

# Run Flask
CMD ["python3", "app.py"]
