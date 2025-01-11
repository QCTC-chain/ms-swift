# base image
FROM python:3.10-slim-bookworm AS base

# production stage
FROM base AS production

EXPOSE 7860

# set timezone
ENV TZ=UTC

WORKDIR /ms-swift

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ libc-dev libffi-dev libgmp-dev libmpfr-dev libmpc-dev

# Copy source code
COPY . /ms-swift
RUN pip install -e '.[all]'

ENTRYPOINT ["/bin/bash", "-c", "swift web-ui --lang=zh"]