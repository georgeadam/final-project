import numpy as np
import pandas as pd
from torchvision.transforms import Compose

from src.utils.preprocess import get_numeric_col_indices
from .creation import data_modules
from .data_module import DataModule
from .eager_dataset import EagerDataset
from .transforms import transforms


class MIMICIV(DataModule):
    def __init__(self, data_dir, batch_size, feeder_args, splitter_args):
        super().__init__(data_dir, batch_size, feeder_args, splitter_args)

        self.data_wrapper = EagerDataset
        self._numeric_cols = None
        self.setup(None)

    def update_train_transform(self, x):
        scaler = transforms.create("scaler", cols=self._numeric_cols)
        scaler.fit(x)

        tensor = transforms.create("tensor")

        self.train_transform = Compose([scaler, tensor])
        self.train_target_transform = None

    def update_inference_transform(self, x):
        scaler = transforms.create("scaler", cols=self._numeric_cols)
        scaler.fit(x)

        tensor = transforms.create("tensor")

        self.inference_transform = Compose([scaler, tensor])
        self.inference_target_transform = None

    def load_data(self):
        data = pd.read_csv(self.data_dir)
        data = data.dropna()

        features = ['HeartRate_Min', 'HeartRate_Max', 'HeartRate_Mean', 'SysBP_Min', 'SysBP_Max', 'SysBP_Mean',
                    'DiasBP_Min', 'DiasBP_Max', 'DiasBP_Mean', 'MeanBP_Min', 'MeanBP_Max', 'MeanBP_Mean',
                    'RespRate_Min', 'RespRate_Max', 'RespRate_Mean', 'TempC_Min', 'TempC_Max', 'TempC_Mean',
                    'SpO2_Min', 'SpO2_Max', 'SpO2_Mean', 'Glucose_Min', 'Glucose_Max', 'Glucose_Mean', 'ANIONGAP',
                    'ALBUMIN', 'BICARBONATE', 'BILIRUBIN', 'CREATININE', 'CHLORIDE', 'GLUCOSE', 'HEMATOCRIT',
                    'HEMOGLOBIN', 'LACTATE', 'MAGNESIUM', 'PHOSPHATE', 'PLATELET', 'POTASSIUM', 'PTT', 'INR', 'PT',
                    'SODIUM', 'BUN', 'WBC', 'age', 'gender', 'ethnicity']

        x = data[features]
        x["age"] = x["age"].map(
            {"0 - 10": 5, "10 - 20": 15, "20 - 30": 25, "30 - 40": 35, "40 - 50": 45, "50 - 60": 55, "60 - 70": 65,
             "70 - 80": 75, "> 80": 85})
        x["age"] = x["age"].astype("category")

        subgroup_cols = ["age", "gender", "ethnicity"]
        self._subgroup_features = x[subgroup_cols]

        x = pd.get_dummies(x)
        self._numeric_cols = get_numeric_col_indices(x)

        y = data["mort_icu"].astype(int)

        x = x.to_numpy()
        y = y.to_numpy()

        x = x.astype("float32")
        y = y.astype(int)

        return x, y

    def set_stats(self, x, y):
        self._data_dimension = x.shape[1]
        self._num_classes = len(np.unique(y))


data_modules.register_builder("mimic_iv", MIMICIV)
