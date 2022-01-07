"""
Lambda wrapper
"""

import json
import logging

from omegaconf import DictConfig

from src.inference import EWSONNXPredictor
from src.pl_data.datamodule import MyDataModule

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)

logger.info("Loading the model")

inferencing_instance = EWSONNXPredictor("./models/model.onnx")
inferencing_instance.set_up(
    datamodule=MyDataModule(
        datasets=None, num_workers=DictConfig({"test": 4}),
        batch_size=16,
        max_length=512, tokenizer="google/bert_uncased_L-2_H-512_A-8"
    )
)


def lambda_handler(event, context):
    """Lambda function handler for predicting discourse type of the given
    sentence

    Args:
        event:
        context:

    Returns:

    """
    if "resource" in event.keys():
        body = event["body"]
        body = json.loads(body)
        logger.info(f"Got the input: {body['sentence']}")

        prediction = inferencing_instance.predict(body["sentence"])
        logger.info(json.dumps(str(prediction)))
        response = {
            "statusCode": 200,
            "headers": {},
            "body": json.dumps(prediction)
        }
        return response
    else:
        logger.info(f"Got the input: {event['sentence']}")
        prediction = inferencing_instance.predict(event["sentence"])
        logger.info(json.dumps(str(prediction)))
        return prediction


def main():
    sentence = "In this situation they need someone to guide them " \
               "through the online course, or they can take a home tutour."
    test = {"sentence": sentence}

    lambda_handler(test, None)


if __name__ == "__main__":
    main()
