"""
Lambda wrapper
"""

import json

import hydra
import omegaconf
from omegaconf import DictConfig

from src.inference import EWSONNXPredictor
from src.pl_data.datamodule import MyDataModule

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
    print(event)

    if "resource" in event.keys():
        body = event["body"]
        body = json.loads(body)
        print(f"Got the input: {body['sentence']}")
        response = inferencing_instance.predict(body["sentence"])
        return {
            "statusCode": 200,
            "headers": {},
            "body": json.dumps(response)
        }
    else:
        return inferencing_instance.predict(event["sentence"])


def main():
    sentence = "In this situation they need someone to guide them " \
               "through the online course, or they can take a home tutour."
    test = {"sentence": sentence}

    lambda_handler(test, None)


if __name__ == "__main__":
    main()
