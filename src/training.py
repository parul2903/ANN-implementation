from src.utils.common_utils import read_config
from src.utils.data_management import get_data
from src.utils.model import create_model, save_model, save_plot
from src.utils.callbacks import get_callbacks
import argparse
import os
import pandas as pd


def training(config_path):
    config = read_config(config_path)

    validation_data_size = config['params']['validation_data_size']
    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = get_data(validation_data_size)

    LOSS_FUNCTION = config['params']['loss_function']
    OPTIMIZER = config['params']['optimizer']
    METRICS = config['params']['metrics']
    NUM_CLASSES = config['params']['no_of_classes']
    model = create_model(LOSS_FUNCTION, OPTIMIZER, METRICS, NUM_CLASSES)

    EPOCHS = config['params']['epochs']
    VALIDATION = (x_valid, y_valid)

    CALLBACK_LIST = get_callbacks(config, x_train)

    model_fit = model.fit(x_train, y_train, epochs=EPOCHS, validation_data=VALIDATION, callbacks = CALLBACK_LIST)

    artifacts_dir = config['artifacts']['artifacts_dir']
    model_name = config['artifacts']['model_name']
    model_dir = config['artifacts']['model_dir']
    model_dir_path = os.path.join(artifacts_dir, model_dir)
    os.makedirs(model_dir_path, exist_ok=True)

    save_model(model, model_name, model_dir_path)

    artifacts_dir = config['artifacts']['artifacts_dir']
    plot_name = config['artifacts']['plot_name']
    plot_dir = config['artifacts']['plot_dir']
    plot_dir_path = os.path.join(artifacts_dir, plot_dir)
    os.makedirs(plot_dir_path, exist_ok=True)
    df = pd.DataFrame(model_fit.history)

    save_plot(df, plot_name, plot_dir_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # we can either use "--config" or "-c"
    parser.add_argument("--config", "-c", default="config.yaml", help="to read the config file")

    parsed_args = parser.parse_args()

    training(config_path = parsed_args.config)
