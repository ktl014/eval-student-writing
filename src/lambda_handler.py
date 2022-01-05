"""
Lambda wrapper
"""

import hydra
import omegaconf
import json

from src.common.utils import PROJECT_ROOT
from src.inference import EWSONNXPredictor

inferencing_instance = EWSONNXPredictor("./models/model.onnx")


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
        http_method = event["httpMethod"]
        if http_method == "GET":
            sentence = event["queryStringParameters"]["sentence"]
            return inferencing_instance.predict(sentence)
        elif http_method == "POST":
            body = json.loads(event["body"])
            return inferencing_instance.predict(body["sentence"])
    else:
        return inferencing_instance.predict(event["sentence"])


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    sentence = "In this situation they need someone to guide them " \
               "through the online course, or they can take a home tutour."
    test = {"sentence": sentence}
    inferencing_instance.set_up(hydra.utils.instantiate(cfg.data.datamodule))

    lambda_handler(test, None)



if __name__ == "__main__":
    main()
