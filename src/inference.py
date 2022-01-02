"""

Usage

>>> python src/inference.py

"""
import logging

import hydra
import omegaconf
import torch
from tabulate import tabulate

from src.common.utils import PROJECT_ROOT
from src.pl_modules.model import MyModel
from src.common.constants import GenericConstants as gc

logger = logging.getLogger(__name__)

class ColaPredictor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = MyModel.load_from_checkpoint(model_path)
        self.model.eval()
        self.model.freeze()
        self.softmax = torch.nn.Softmax(dim=1)
        #todo include into the config data the list of labels
        self.labels = ["Lead", "Position", "Evidence", "Claim",
                       "Concluding Statement", "Counterclaim",
                       "Rebuttal"]

    def set_up(self, datamodule):
        self.processor = datamodule

    def predict(self, text):
        inference_sample = {gc.SENTENCE: text}
        processed = self.processor.tokenize(inference_sample)
        logits = self.model(
            torch.tensor([processed["input_ids"]]),
            torch.tensor([processed["attention_mask"]]),
        )
        scores = self.softmax(logits[0]).tolist()[0]
        predictions = []
        for score, label in zip(scores, self.labels):
            predictions.append({"label": label, "score": score})
        return predictions


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    sentence = "In this situation they need someone to guid them through the online course, or they can take a home tutour."
    root_dir = hydra.utils.get_original_cwd()
    model_path = f"{root_dir}/models/best-checkpoint.ckpt"
    predictor = ColaPredictor(model_path=model_path)
    predictor.set_up(
        datamodule=hydra.utils.instantiate(cfg.data.datamodule)
    )
    logger.info(f"MODEL_LOADED: {model_path}")
    logger.info(f"INPUT: {sentence}")
    logger.info(f"OUTPUT (see below):\n{tabulate(predictor.predict(sentence))}")


if __name__ == "__main__":
    main()
