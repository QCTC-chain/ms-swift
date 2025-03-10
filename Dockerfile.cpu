# base image
FROM vllm-cpu-swfit-v2:v0.7.0 

# set timezone
ENV TZ=UTC

WORKDIR /ms-swift
# Copy source code
COPY . /ms-swift
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -e '.[all]'

EXPOSE 7860

ENTRYPOINT ["/bin/bash", "-c", "swift web-ui --lang=zh"]