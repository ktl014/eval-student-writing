"""
Lambda wrapper
"""

from src.inference import EWSONNXPredictor

inferencing_instance = EWSONNXPredictor("./models/model.onnx")


def lambda_handler(event, context):
    """Lambda function handler for predicting linguistic acceptability of
	the given sentence

    Args:
        event:
        context:

    Returns:

    """
    print(event)
    return inferencing_instance.predict(event["sentence"])


if __name__ == "__main__":
    sentence = "In this situation they need someone to guid them " \
               "through the online course, or they can take a home tutour."
    test = {"sentence": sentence}
    lambda_handler(test, None)
