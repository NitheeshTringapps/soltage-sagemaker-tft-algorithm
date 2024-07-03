import os
import pandas as pd
import numpy as np
import random
import torch
import lightning.pytorch as pl
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_forecasting.data.encoders import NaNLabelEncoder
import json

import warnings
warnings.filterwarnings("ignore")  # avoid printing out absolute paths

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_data(input_dir):
    return pd.read_csv(os.path.join(input_dir, 'train.csv'))

def train(train_start_dt, train_end_dt, prediction_start_dt, prediction_end_dt):
    set_seed(42)

    input_dir = '/opt/ml/input/data/training'
    output_dir = '/opt/ml/model'

    all_data_merged_df = load_data(input_dir)
    train_start_dt = pd.to_datetime(train_start_dt)
    train_end_dt = pd.to_datetime(train_end_dt)
    prediction_start_dt = pd.to_datetime(prediction_start_dt)
    prediction_end_dt = pd.to_datetime(prediction_end_dt)

    data = all_data_merged_df.copy()
    data['DateTime'] = pd.to_datetime(data['DateTime'])
    data = data[(data['DateTime'] >= train_start_dt)]
    data['time_idx'] = (data['DateTime'] - data['DateTime'].min()) / pd.Timedelta(1, 'H')
    data['time_idx'] = data['time_idx'].astype(int)

    all_data = data.copy()
    data = data[(data['DateTime'] >= train_start_dt) & (data['DateTime'] <= train_end_dt)]

    max_prediction_length = 24  # last 24 hours
    max_encoder_length = 24
    training_cutoff = data["time_idx"].max() - max_prediction_length

    training = TimeSeriesDataSet(
        data[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="Load",
        group_ids=["RegionCode"],
        min_encoder_length=max_encoder_length // 2,  # keep encoder length long (as it is in the validation set)
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=["RegionCode"],
        time_varying_known_categoricals=["Condition", "DayOfWeek", "DayType", "HolidayType"],
        time_varying_known_reals=["time_idx", "is_day", "Temperature", "DewPoint", "Humidity", "Pressure", "WindSpeed"],
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=["Load"],
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,  # allow time series with missing time steps
        categorical_encoders={
            'Condition': NaNLabelEncoder(add_nan=True),
            'DayOfWeek': NaNLabelEncoder(add_nan=True),
            'DayType': NaNLabelEncoder(add_nan=True),
            'HolidayType': NaNLabelEncoder(add_nan=True),
        },
    )

    validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)
    batch_size = 128  # set this between 32 to 128
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
    lr_logger = LearningRateMonitor()  # log the learning rate
    logger = TensorBoardLogger("lightning_logs")  # logging results to a tensorboard

    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="cpu",
        enable_model_summary=True,
        gradient_clip_val=0.156141751871966,
        callbacks=[lr_logger, early_stop_callback],
        logger=logger,
    )

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.0197534589190262,
        hidden_size=45,
        attention_head_size=4,
        dropout=0.242742470997258,
        hidden_continuous_size=14,
        loss=QuantileLoss(),
        reduce_on_plateau_patience=4,
    )
    print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    # Save the model
    tft_path = os.path.join(output_dir, 'model')
    tft.save(tft_path)

    # Save the best model according to the validation loss
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
    best_tft_path = os.path.join(output_dir, 'best_model')
    best_tft.save(best_tft_path)
    print(f"Best model saved to: {best_tft_path}")

if __name__ == '__main__':
    prefix = '/opt/ml/'
    param_path = os.path.join(prefix, 'input/config/hyperparameters.json')
    with open(param_path, 'r') as tc:
        trainingParams = json.load(tc)
    print("[INFO] Hyperparameters: ", trainingParams)

    print(f"TRAIN_START_DT: {trainingParams['train_start_dt']}")
    print(f"TRAIN_END_DT: {trainingParams['train_end_dt']}")
    print(f"PREDICTION_START_DT: {trainingParams['prediction_start_dt']}")
    print(f"PREDICTION_END_DT: {trainingParams['prediction_end_dt']}")

    # Call the training function
    train(trainingParams['train_start_dt'], trainingParams['train_end_dt'], trainingParams['prediction_start_dt'], trainingParams['prediction_end_dt'])
