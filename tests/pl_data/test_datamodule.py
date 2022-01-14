import os

import hydra
import pandas as pd
import pytest
from hydra import compose

from src.common.constants import GenericConstants as gc
from src.common.utils import PROJECT_ROOT

os.chdir(PROJECT_ROOT)

cfg = compose(config_name="default")


class TestDataModule:
    @pytest.fixture
    def data_module(self):
        return hydra.utils.instantiate(cfg.data.datamodule, _recursive_=False)

    def test_prepare_data(self, data_module):
        # === Expected Output ===#

        # Load dataset into dataframe from hydra config
        train_df = pd.read_csv(data_module.datasets.train.path)
        row_count = train_df.shape[0]

        # Count 80% dataset = train_dataset_size
        train_dataset_size = round(row_count * 80 / 100)

        # Count 20% dataset = val_dataset_size
        val_dataset_size = round(row_count * 20 / 100)

        # Count number of unique labels
        num_unique_labels = train_df[gc.LABEL].nunique()

        # === Trigger output ===#
        data_module.prepare_data()

        # Check if prepare_data() correctly split dataset into 80-train, 20-val
        assert len(data_module.train_datasets) == train_dataset_size
        assert len(data_module.val_datasets) == val_dataset_size

        # Check if number of labels are correctly computed from the dataset
        assert data_module.labels.num_classes == num_unique_labels

    @pytest.mark.xfail(reason="Has not yet implemented")
    def test_tokenize_and_label_encoding(self, data_module):
        # === Input ===#
        data = {
            "discourse_test": ["This is a dummy sentence", "Hello World!"],
            "discourse_type": ["dummy", "friendly"],
        }
        df = pd.DataFrame(data=data)

        # === Expected Output ===#
        tokens = data_module.tokenize_and_label_encoding(df)
        print(tokens)
