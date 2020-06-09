# vim: set filetype=dockerfile :

# Base image with Apex
FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-devel as base
MAINTAINER tidalpaladin

RUN apt-get update && apt-get install -y git

# install nvidia apex
ADD ./docker/common/install_apex.sh install_apex.sh
RUN bash ./install_apex.sh
RUN rm install_apex.sh

# cv2 dependencies
RUN apt-get -y install libglib2.0-0 libsm6 libxrender-dev libxext6

VOLUME ["/app/data", "/app/outputs", "/app/conf"]

COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt

# copy and install combustion
COPY combustion/requirements.txt /app/combustion/
WORKDIR /app
RUN pip install -r /app/combustion/requirements.txt
RUN pip install -U git+https://github.com/facebookresearch/hydra.git
COPY combustion /app/combustion
RUN pip install -e /app/combustion


# Release version
FROM base as release
ARG PROJECT
COPY $PROJECT /app/$PROJECT

USER 1000
ENTRYPOINT ["bash"]

# Development version
FROM base as dev
RUN pip install -e .[dev]

ARG PROJECT
COPY src/$PROJECT /app/src/$PROJECT
COPY ./tests /app/tests

USER 1000
ENTRYPOINT ["bash"]
CMD ["python", "-m", "pytest", "-n", "auto", "--dist=loadfile", "-s", "-v", "/app/tests/"]

