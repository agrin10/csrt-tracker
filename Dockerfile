FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /usr/local/app
COPY requirements.txt .
# Install the application dependencies
RUN --mount=type=cache,target=~/.cache/pip \
	--mount=type=bind,source=./requirements.txt,target=/app/requirements.txt

# Copy in the source code

# Pre-download ResNet50 into the torch.hub cache
# so that at runtime we never do an HTTP fetch
RUN python3 - <<EOF
import torch
# This will pull both the repo and the weights into /root/.cache/torch/hub
torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
EOF
COPY . .


CMD ["python3", "app.py"]