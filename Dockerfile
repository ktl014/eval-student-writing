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

WORKDIR ./
ENV PROJECT_ROOT=./
ENV WANDB_DIR=$WANDB_DIR
ENV WANDB_CONFIG_DIR=$WANDB_CONFIG_DIR
ENV WANDB_CACHE_DIR=$WANDB_CACHE_DIR
=======
ENV PROJECT_ROOT=/app

RUN apt-get -y update && apt-get install -y libzbar-dev

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# pulling the trained model
RUN dvc pull models/model.onnx.dvc

ENV PYTHONPATH "${PYTHONPATH}:./"
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# pulling the trained model
RUN dvc pull models/model.onnx.dvc

RUN python -m lambda_handler
RUN chmod -R 0755 $MODEL_DIR

CMD ["lambda_handler.lambda_handler"]
