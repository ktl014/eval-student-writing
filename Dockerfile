FROM public.ecr.aws/lambda/python:3.8

ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG MODEL_DIR=./models
ARG WANDB_DIR=./wandb_dir
ARG WANDB_CONFIG_DIR=./wandb_config_dir
ARG WANDB_CACHE_DIR=./wandb_cache_dir
RUN mkdir $MODEL_DIR
RUN mkdir -p $WANDB_DIR/wandb
RUN mkdir $WANDB_CONFIG_DIR
RUN mkdir $WANDB_CACHE_DIR


ENV TRANSFORMERS_CACHE=$MODEL_DIR \
    TRANSFORMERS_VERBOSITY=error

ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
    AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY

RUN yum install git -y && yum -y install gcc-c++

COPY requirements.txt requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt --no-cache-dir

COPY ./ ./
COPY src/lambda_handler.py ./
ENV PROJECT_ROOT=./
ENV WANDB_DIR=$WANDB_DIR
ENV WANDB_CONFIG_DIR=$WANDB_CONFIG_DIR
ENV WANDB_CACHE_DIR=$WANDB_CACHE_DIR

ENV PYTHONPATH "${PYTHONPATH}:./"
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# pulling the trained model
RUN dvc pull models/model.onnx.dvc

# RUN chmod -R 0755 $WANDB_DIR
# RUN chmod -R 0755 $WANDB_CONFIG_DIR
# RUN chmod -R 0755 $WANDB_CACHE_DIR

# RUN chmod -R 0755 ./
RUN ls

RUN python -m lambda_handler
RUN chmod -R 0755 $MODEL_DIR
# RUN chmod -R 0755 $WANDB_DIR/wandb
CMD ["lambda_handler.lambda_handler"]

#FROM python:3.8-slim-buster
#COPY ./ /app
#WORKDIR /app
#ARG AWS_ACCESS_KEY_ID
#ARG AWS_SECRET_ACCESS_KEY
#ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
#    AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
#ENV PROJECT_ROOT=/app
#RUN apt-get -y update && apt-get install -y libzbar-dev
#RUN pip install --upgrade pip
#RUN pip install -r requirements.txt
#
## pulling the trained model
#RUN dvc pull models/model.onnx.dvc
#
#ENV PYTHONPATH "${PYTHONPATH}:./"
#ENV LC_ALL=C.UTF-8
#ENV LANG=C.UTF-8
#EXPOSE 8000
#CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]