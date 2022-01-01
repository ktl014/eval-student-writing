import os
import sys
from pathlib import Path

import hydra
import pytest
import torch
import torchmetrics
from hydra import compose

from src.common.constants import GenericConstants as gc
# from src.common.utils import PROJECT_ROOT

# print(str(Path(__file__).resolve().parents[0]))
# sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
# sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
#
# os.chdir(PROJECT_ROOT)

cfg = compose(config_name="default")


class TestModel:
    @pytest.fixture
    def data_module(self):
        data_module = hydra.utils.instantiate(cfg.data.datamodule,
                                              _recursive_=False)
        data_module.prepare_data()
        data_module.setup()
        return data_module

    @pytest.fixture
    def model(self):
        # No need to use GPU or multi-processing on unit test
        cfg.train.pl_trainer.gpus = 0
        cfg.data.datamodule.num_workers.train = 0
        cfg.data.datamodule.num_workers.val = 0
        cfg.data.datamodule.num_workers.test = 0

        # Switch wandb mode to offline to prevent online logging
        cfg.logging.wandb.mode = "offline"

        return hydra.utils.instantiate(
            cfg.model.modelmodule,
            optim=cfg.optim,
            data=cfg.data,
            logging=cfg.logging,
            _recursive_=False,
        )

    def test_unit(self, model):
        # === Input ===#
        model_name = cfg.model.modelmodule.model_name
        num_labels = cfg.model.modelmodule.num_labels

        # Check if number of labels are correctly save within the model
        assert model.num_classes == num_labels

        # Check if model is loaded
        assert model.bert.name_or_path == model_name

        # Check if metrics are initialized
        assert type(model.train_accuracy_metric) == torchmetrics.Accuracy()
        assert type(model.val_accuracy_metric) == torchmetrics.Accuracy()
        assert type(model.f1_metric) == torchmetrics.F1(num_classes=num_labels)
        assert type(model.precision_macro_metric) == torchmetrics.Precision(
            average="macro", num_classes=num_labels
        )
        assert isinstance(model.recall_macro_metric,
                          type(torchmetrics.Recall(average="macro",
                                                   num_classes=num_labels)))
        assert isinstance(model.precision_micro_metric,
                          type(torchmetrics.Precision(average="micro")))
        assert isinstance(model.recall_micro_metric,
                          type(torchmetrics.Recall(average="micro")))

    def test_foward(self, data_module, model):
        # === Input ===#
        batch = dict()
        train_dataloader = next(iter(data_module.train_dataloader()))
        batch["input_ids"] = train_dataloader["input_ids"]
        batch["attention_mask"] = train_dataloader["attention_mask"]
        batch[gc.LABEL] = train_dataloader[gc.LABEL]

        # === Trigger Output ===#
        outputs = model.forward(
            batch["input_ids"], batch["attention_mask"], batch[gc.LABEL]
        )
        logits = outputs.logits
        loss = outputs.loss

        # Check for type and shape of output.logits
        assert type(logits) == torch.Tensor
        assert logits.device == torch.device(type="cpu")
        assert logits.dtype == torch.float32
        assert logits.shape[0] == cfg.data.datamodule.batch_size.train
        assert logits.shape[1] == cfg.model.modelmodule.num_labels

        # Check if loss is computed correctly
        assert type(loss) == torch.Tensor
        assert loss.device == torch.device(type="cpu")
        assert loss.dtype == torch.float32
        assert loss.detach().numpy() >= 0
