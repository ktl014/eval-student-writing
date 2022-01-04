"""

Usage

>>> python src/inference.py

"""
import logging
import time
from functools import wraps

import hydra
import numpy as np
import omegaconf
import onnxruntime as ort
import torch
from scipy.special import softmax
from tabulate import tabulate

from src.common.constants import GenericConstants as gc
from src.common.utils import PROJECT_ROOT
from src.pl_modules.model import MyModel

logger = logging.getLogger(__name__)


def timing(f):
    """Decorator for timing functions
    Usage:
    @timing
    def function(a):
        pass
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        logger.debug("function:%r took: %2.5f sec" % (f.__name__, end - start))
        return result

    return wrapper


class BasePredictor:
    def __init__(self):
        self.labels = [
            "Lead",
            "Position",
            "Evidence",
            "Claim",
            "Concluding Statement",
            "Counterclaim",
            "Rebuttal",
        ]
        self.inference_sample = {gc.SENTENCE: None}

    def set_up(self, datamodule):
        self.processor = datamodule


class EWSPredictor(BasePredictor):
    def __init__(self, model_path):
        super().__init__()
        self.model_path = model_path
        self.model = MyModel.load_from_checkpoint(model_path)
        self.model.eval()
        self.model.freeze()
        self.softmax = torch.nn.Softmax(dim=1)

    def predict(self, text):
        self.inference_sample[gc.SENTENCE] = text
        processed = self.processor.tokenize(self.inference_sample)
        logits = self.model(
            torch.tensor([processed["input_ids"]]),
            torch.tensor([processed["attention_mask"]]),
        )
        scores = self.softmax(logits[0]).tolist()[0]
        predictions = []
        for score, label in zip(scores, self.labels):
            predictions.append({"label": label, "score": score})
        return predictions


class EWSONNXPredictor(BasePredictor):
    def __init__(self, model_path):
        super().__init__()
        self.ort_session = ort.InferenceSession(model_path)

    @timing
    def predict(self, text):
        self.inference_sample[gc.SENTENCE] = text
        processed = self.processor.tokenize(self.inference_sample)
        ort_inputs = {
            "input_ids": np.expand_dims(processed["input_ids"],
                                        axis=0
                                        ).astype("int64"),
            "attention_mask": np.expand_dims(processed["attention_mask"],
                                             axis=0
                                             ).astype("int64"),
        }

        ort_outs = self.ort_session.run(None, ort_inputs)
        scores = softmax(ort_outs[0])[0]
        predictions = []
        for score, label in zip(scores, self.labels):
            predictions.append({"label": label, "score": score})
        return predictions


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    sentence = "In this situation they need someone to guid them " \
               "through the online course, or they can take a home tutour."
    root_dir = hydra.utils.get_original_cwd()

    # Instantiate predictor
    # Predictor will depend on configuration if onnx model
    # is selected. We recommend onnx when setting up a microservice
    # for the given model.
    if cfg.model.modelmodule.onnx:
        model_path = f"{root_dir}/models/model.onnx"
        predictor = EWSONNXPredictor(model_path=model_path)
    else:
        model_path = f"{root_dir}/models/best-checkpoint.ckpt"
        predictor = EWSPredictor(model_path=model_path)

    # Set up the tokenizer for the model
    predictor.set_up(datamodule=hydra.utils.instantiate(cfg.data.datamodule))

    # Run predictions
    logger.info(f"MODEL_LOADED: {model_path}")
    logger.info(f"INPUT: {sentence}")
    logger.info(f"OUTPUT (see below):\n"
                f"{tabulate(predictor.predict(sentence))}")


if __name__ == "__main__":
    main()
