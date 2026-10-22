# One image for the competition: it carries the environment, the two programs
# and the benchopt benchmark (/compet/benchmark). The phase bundle mounted on
# /app/input_data only carries the run config. Build from the repo root:
#
#     docker build -f tools/Dockerfile -t <dockerhub-user>/<competition>:v1 .
#
# The same image serves the Codabench workers and the participants:
#
#     # stage the data into a host folder (bind-mounted read-only as
#     # /app/data for the evaluation runs)
#     docker run -v <host-data>:/app/data \
#         <image> benchopt prepare /compet/benchmark -d Real
#
#     # reproduce the platform evaluation on a submission folder
#     docker run --gpus all -v <host-data>:/app/data \
#         -v <phase>/input_data:/app/input_data \
#         -v <submission>:/app/ingested_program -v <results>:/app/output \
#         <image> python3 /compet/ingestion_program/ingestion.py
#
# Python 3.12 base; torch comes from the PyTorch wheel index, whose wheels
# bundle the CUDA runtime, so GPU works through the nvidia container runtime
# without a CUDA base image.
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive

# torch with an explicit CUDA channel: cu126 runs on any R525+ driver (cu128+
# for Blackwell GPUs, cpu for CI). Installed first so requirements.txt keeps
# this build.
ARG TORCH_CHANNEL=cu126
RUN pip install -U pip && pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/${TORCH_CHANNEL}

# All Python dependencies a submission may import: nothing is installed at
# submission time.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# The programs + the benchopt benchmark they run (baked in, so the phase
# bundle only carries config; an image tag pins the exact evaluation).
COPY ingestion_program /compet/ingestion_program
COPY scoring_program /compet/scoring_program
COPY benchmark /compet/benchmark

# Data is bind-mounted read-only at /app/data by the compute worker; the
# phase config lands on /app/input_data.
ENV BENCHOPT_DATA_HOME=/app/data
VOLUME /app/data

# Fail the build if the stack is incoherent.
RUN python -c "import benchopt, pandas, sklearn, torch"
