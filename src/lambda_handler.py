"""
Lambda wrapper
"""

import json

import hydra
import omegaconf

from src.common.utils import PROJECT_ROOT
from src.inference import EWSONNXPredictor

inferencing_instance = EWSONNXPredictor("./models/model.onnx")


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def init_datamodule(cfg: omegaconf.DictConfig):
    inferencing_instance.set_up(hydra.utils.instantiate(cfg.data.datamodule))


def lambda_handler(event, context):
    """Lambda function handler for predicting discourse type of the given
    sentence

    Args:
        event:
        context:

    Returns:

    """
    init_datamodule()
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
