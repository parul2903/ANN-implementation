from src.utils.common_utils import read_config
from src.utils.data_management import get_data 
from src.utils.model import create_model
import argparse

def training(config_path):
    config = read_config(config_path)

    validation_datasize = config['params']['validation_data_size']
    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = get_data(validation_datasize)

    LOSS_FUNCTION = config['params']['loss_function']
    OPTIMIZER = config['params']['optimizer']
    METRICS = config['params']['metrics']
    NUM_CLASSES = config['params']['no_of_classes']
    model = create_model(LOSS_FUNCTION, OPTIMIZER, METRICS, NUM_CLASSES)

    EPOCHS = config['params']['epochs']
    VALIDATION = (x_valid, y_valid)

    model_fit = model.fit(x_train, y_train, epochs = EPOCHS, validation_data = VALIDATION)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # we can either use "--config" or "-c"
    parser.add_argument("--config", "-c", default = "config.yaml", help = "to read the config file")

    parsed_args = parser.parse_args()

    training(config_path = parsed_args.config) 
