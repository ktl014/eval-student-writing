from typing import Any, Dict, Sequence, Tuple, Union

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
import torchmetrics
from torch.optim import Optimizer
from transformers import AutoModelForSequenceClassification

import wandb
from src.common.constants import GenericConstants as gc
from src.common.utils import PROJECT_ROOT


class MyModel(pl.LightningModule):
    def __init__(self, model_name, num_labels, *args, **kwargs) -> None:
        super().__init__()

        # populate self.hparams with args and kwargs automagically!
        self.save_hyperparameters()
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        self.num_classes = num_labels

        # Initialize metrics
        self.train_accuracy_metric = torchmetrics.Accuracy()
        self.val_accuracy_metric = torchmetrics.Accuracy()
        self.f1_metric = torchmetrics.F1(num_classes=self.num_classes)
        self.precision_macro_metric = torchmetrics.Precision(
            average="macro", num_classes=self.num_classes
        )
        self.recall_macro_metric = torchmetrics.Recall(
            average="macro", num_classes=self.num_classes
        )
        self.precision_micro_metric = torchmetrics.Precision(average="micro")
        self.recall_micro_metric = torchmetrics.Recall(average="micro")

    def forward(
        self, input_ids, attention_mask, labels=None
    ) -> Dict[str, torch.Tensor]:
        outputs = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        return outputs

    def step(self, batch: Any, batch_idx: int):
        outputs = self.forward(
            batch["input_ids"], batch["attention_mask"], labels=batch[gc.LABEL]
        )
        preds = torch.argmax(outputs.logits, 1)
        return preds, outputs.logits, outputs.loss

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        # Conduct forward step and retrieve
        # loss and logits output
        labels = batch[gc.LABEL]
        preds, logits, loss = self.step(batch, batch_idx)

        # Calculate metrics
        train_acc = self.train_accuracy_metric(preds, labels)

        # Log metrics
        self.log("train/loss", loss, prog_bar=True, on_epoch=True)
        self.log("train/acc", train_acc, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        labels = batch[gc.LABEL]
        preds, logits, loss = self.step(batch, batch_idx)

        # Metrics
        valid_acc = self.val_accuracy_metric(preds, labels)
        precision_macro = self.precision_macro_metric(preds, labels)
        recall_macro = self.recall_macro_metric(preds, labels)
        precision_micro = self.precision_micro_metric(preds, labels)
        recall_micro = self.recall_micro_metric(preds, labels)
        f1 = self.f1_metric(preds, labels)

        # Logging metrics
        self.log("valid/loss", loss, prog_bar=True, on_step=True)
        self.log("valid/acc", valid_acc, prog_bar=True, on_epoch=True)
        self.log("valid/precision_macro", precision_macro,
                 prog_bar=True, on_epoch=True)
        self.log("valid/recall_macro", recall_macro,
                 prog_bar=True, on_epoch=True)
        self.log("valid/precision_micro", precision_micro,
                 prog_bar=True, on_epoch=True)
        self.log("valid/recall_micro", recall_micro,
                 prog_bar=True, on_epoch=True)
        self.log("valid/f1", f1, prog_bar=True, on_epoch=True)
        return {"labels": labels, "logits": logits}

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss = self.step(batch, batch_idx)
        self.log_dict(
            {"test_loss": loss},
        )
        return loss

    def validation_epoch_end(self, outputs):
        labels = torch.cat([x["labels"] for x in outputs])
        logits = torch.cat([x["logits"] for x in outputs])
        # preds = torch.argmax(logits, 1)

        # There are multiple ways to track the metrics
        # 1. Confusion matrix plotting using inbuilt W&B method
        self.logger.experiment.log(
            {
                "conf": wandb.plot.confusion_matrix(
                    probs=logits.cpu().numpy(), y_true=labels.cpu().numpy()
                )
            }
        )

        # 2. Confusion Matrix plotting using scikit-learn method
        # wandb.log({"cm": wandb.sklearn.plot_confusion_matrix(labels.numpy(),
        # preds)})

        # 3. Confusion Matric plotting using Seaborn
        # data = confusion_matrix(labels.numpy(), preds.numpy())
        # df_cm = pd.DataFrame(data, columns=np.unique(labels),
        # index=np.unique(labels))
        # df_cm.index.name = "Actual"
        # df_cm.columns.name = "Predicted"
        # plt.figure(figsize=(7, 4))
        # plot = sns.heatmap(
        #     df_cm, cmap="Blues", annot=True, annot_kws={"size": 16}
        # )  # font size
        # self.logger.experiment.log({"Confusion Matrix": wandb.Image(plot)})

        self.logger.experiment.log(
            {"roc": wandb.plot.roc_curve(labels.cpu().numpy(),
                                         logits.cpu().numpy())}
        )

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Tuple[Sequence[Optimizer], Sequence[Any]]]:
        """
        Choose what optimizers and learning-rate schedulers to
        use in your optimization.

        Normally you'd need one. But in the case of GANs or similar
        you might have multiple.

        Return:
            Any of these 6 options.
            - Single optimizer.
            - List or Tuple - List of optimizers.
            - Two lists - The first list has multiple optimizers, the second
              a list of LR schedulers (or lr_dict).
            - Dictionary, with an 'optimizer' key, and (optionally) a
              'lr_scheduler'
              key whose value is a single LR scheduler or lr_dict.
            - Tuple of dictionaries as described, with an optional
              'frequency' key.
            - None - Fit will run without any optimizer.
        """
        opt = hydra.utils.instantiate(
            self.hparams.optim.optimizer,
            params=self.parameters(),
            _convert_="partial"
        )
        if not self.hparams.optim.use_lr_scheduler:
            return [opt]
        scheduler = hydra.utils.instantiate(
            self.hparams.optim.lr_scheduler, optimizer=opt
        )
        return [opt], [scheduler]


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    model: pl.LightningModule = hydra.utils.instantiate(
        cfg.model.modelmodule,
        optim=cfg.optim,
        data=cfg.data,
        logging=cfg.logging,
        _recursive_=False,
    )
    print("Success!") if model else print("Fail!")


if __name__ == "__main__":
    main()
