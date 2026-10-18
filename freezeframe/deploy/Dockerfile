# Enclave image for LinuxONE (s390x) or x86_64 Linux. Snap ML publishes manylinux wheels for both.
# Build on the target:   docker build -f deploy/Dockerfile -t freezeframe .
# In production this image runs inside IBM Confidential Computing Container Runtime
# (Hyper Protect Virtual Servers) on Secure Execution; see deploy/linuxone.md.
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml ./
COPY freezeframe ./freezeframe
COPY bench ./bench
COPY demo ./demo
RUN pip install --no-cache-dir -e '.[xgb]'
# data/ and artifacts/ are mounted at run time, never baked into the image
VOLUME ["/app/data", "/app/artifacts", "/app/results"]
ENV PORT=8443
EXPOSE 8443
CMD ["python", "-m", "freezeframe.enclave.service"]
