# The competition image: a base image plus requirements.txt. Build it from
# the repo root and push it, then set docker_image in competition.yaml:
#
#     docker build -f tools/Dockerfile -t <dockerhub-user>/<competition>:v1 .
#     # GPU (self-hosted GPU worker only)
#     docker build -f tools/Dockerfile -t <dockerhub-user>/<competition>:v1 \
#         --build-arg BASE=ghcr.io/x-datascience-datacamp/codabench-benchopt-gpu .
#
# With nothing in requirements.txt beyond the base image, skip this build and
# use the base image itself as docker_image.
ARG BASE=ghcr.io/x-datascience-datacamp/codabench-benchopt
FROM ${BASE}

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
