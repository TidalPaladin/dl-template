# vim: set filetype=dockerfile :
ARG DOCKER_BASE
ARG PROJECT
FROM pytorch/pytorch:$DOCKER_BASE as base

RUN apt-get update && apt-get install -y git 

VOLUME ["/app/data", "/app/outputs", "/app/conf"]

WORKDIR /app

# install dependencies
ADD combustion setup.py requirements.txt /app/
RUN pip3 install -r requirements.txt

# install project
ADD $(PROJECT) /app/
RUN pip3 install -e .

# Release version
FROM base as release
ARG PROJECT

CMD ["python", "-m", "project"]
