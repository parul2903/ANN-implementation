import tensorflow as tf
import numpy as np
import os
import time

def get_timestamp(name):
    timestamp = time.asctime().replace(" ", "_").replace(":", "_")
    unique_name = f"{name}_at_{timestamp}"
    return unique_name

def get_callbacks(config, x_train):

    # TENSORBOARD
    logs = config['logs']
    unique_dir_name = get_timestamp("tb_logs")
    TENSORBOARD_ROOT_LOG_DIR = os.path.join(logs["log_dir"], logs["TENSORBOARD_ROOT_LOG_DIR"], unique_dir_name)

    os.makedirs(TENSORBOARD_ROOT_LOG_DIR, exist_ok = True)
    
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir = TENSORBOARD_ROOT_LOG_DIR)

    file_writer = tf.summary.create_file_writer(logdir = TENSORBOARD_ROOT_LOG_DIR)
    with file_writer.as_default():
        images = np.reshape(x_train[10:30], (-1,28,28,1))
        tf.summary.image("20 handwritten digit samples", images, max_outputs = 25, step = 0)
    
    # EARLY STOPPING
    params = config["params"]
    earlyStopping_cb = tf.keras.callbacks.EarlyStopping(patience = params["patience"], restore_best_weights = params["restore_best_weights"])  

    # MODEL CHECKPOINT
    artifacts = config["artifacts"]
    ckpt_dir = os.path.join(artifacts["artifacts_dir"], artifacts["CHECKPOINTS_DIR"])

    os.makedirs(ckpt_dir, exist_ok = True)

    CKPT_path = os.path.join(ckpt_dir, 'model_ckpt.h5')
    checkpointing_cb = tf.keras.callbacks.ModelCheckpoint(CKPT_path, save_best_only = True)

    return [tensorboard_cb, earlyStopping_cb, checkpointing_cb]

