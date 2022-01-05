FROM python:3.8-slim-buster

COPY ./ /app
WORKDIR /app

ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY

ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
    AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY

ENV PROJECT_ROOT=/app

RUN apt-get -y update && apt-get install -y libzbar-dev

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# RUN dvc remote modify myremote access_key_id $AWS_ACCESS_KEY_ID
# RUN dvc remote modify myremote secret_access_key $AWS_SECRET_ACCESS_KEY

# pulling the trained model
RUN dvc pull models/model.onnx.dvc

ENV PYTHONPATH "${PYTHONPATH}:./"
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
EXPOSE 8000
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
