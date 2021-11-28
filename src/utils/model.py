import tensorflow as tf
import time
import os
import pandas as pd
import matplotlib.pyplot as plt


def create_model(LOSS_FUNCTION, OPTIMIZER, METRICS, NUM_CLASSES):
    LAYERS = [
        tf.keras.layers.Flatten(input_shape=[28, 28], name="inputLayer"),
        tf.keras.layers.Dense(300, activation="relu", name="hiddenLayer1"),
        tf.keras.layers.Dense(100, activation="relu", name="hiddenLayer2"),
        tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="outputLayer")]

    model_clf = tf.keras.models.Sequential(LAYERS)
    model_clf.summary()
    model_clf.compile(loss=LOSS_FUNCTION, optimizer=OPTIMIZER, metrics=METRICS)

    return model_clf


def get_unique_filename(filename):
    unique_filename = time.strftime(f"%Y%m%d_%H%M%S_{filename}")
    return unique_filename


def save_model(model, model_name, model_dir):
    unique_filename = get_unique_filename(model_name)
    model_path = os.path.join(model_dir, unique_filename)
    model.save(model_path)


def save_plot(df, plot_name, plot_dir):
    pd.DataFrame(df).plot(figsize=(10, 7))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)

    unique_plot_name = get_unique_filename(plot_name)
    os.makedirs(plot_dir, exist_ok=True)
    plotPath = os.path.join(plot_dir, unique_plot_name)
    plt.savefig(plotPath)
    plt.show()

