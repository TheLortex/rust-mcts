import argparse
from subprocess import PIPE, DEVNULL
from tqdm import tqdm
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras import optimizers
import tensorflow_addons as tfa
import numpy as np
import pickle
from threading import Thread, RLock
import os
import subprocess
from datetime import datetime
from munch import Munch

import toml

from settings import *
from networks import *
from replay_buffer import ReplayBuffer, BufferThread, AlphaZeroGenerator, MuGenerator

# Allow dynamic memory growth in order not to take all the GPU resource
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--new", help="create new network", action="store_true")
parser.add_argument("--config", type=str, help="config file name")
parser.add_argument("--method", type=str,
                    help="learning method", choices=["alpha", "mu"])
args = parser.parse_args()

# Load config file
config_file = "config/{}.toml".format(args.config)
method = args.method
config = toml.load(config_file)
# Setup default parameters.
if "mu" in config:
    if "reward_support" not in config["mu"]:
        config["mu"]["reward_support"] = 0
    if "value_support" not in config["mu"]["puct"]:
        config["mu"]["puct"]["value_support"] = 0

if "alpha" in config:
    if "value_support" not in config["alpha"]["puct"]:
        config["alpha"]["puct"]["value_support"] = 0

if "game" in config:
    if "history" not in config["game"]:
        config["game"]["history"] = 1

config = Munch.fromDict(config)
base_dir = dir_name(config, method)

print("Loaded config: {}".format(config))

# REPLAY BUFFER
print("Creating replay buffer.")
training_data_path = "./data/{}/training_data/".format(base_dir)
if os.path.exists(training_data_path+"replay_buffer.pkl"):
    print("| Loaded replay buffer")
    f = open(training_data_path+"replay_buffer.pkl", "rb")
    replay_buffer = pickle.load(f)
    f.close()
    print("Status: {} / {}".format(replay_buffer.states_count, replay_buffer.max_index))
else:
    print("| Starting replay buffer from scratch")
    os.makedirs(training_data_path, exist_ok=True)
    replay_buffer = ReplayBuffer(0, 0, 0, [None]*config.training.replay_buffer)

buffer_thr = BufferThread(config, replay_buffer, training_data_path,
                          fifo_path="./data/{}/fifo".format(base_dir))

# Create models and data generators.
if method == "alpha":
    if not hasattr(config, "alpha"):
        print("Alpha is not supported for this game.")
        exit(-1)

    model_path = "./data/{}/model/".format(base_dir)
    if os.path.exists(model_path) and not(args.new):
        print("| Loaded previous instance of the model.")
        network = models.load_model(model_path)
        print(network.summary())
        try:
            start_epoch = np.load(model_path+"epoch.npy")
        except FileNotFoundError:
            start_epoch = 0
        print("Epoch: {}".format(start_epoch))
    else:
        print("| Starting model from scratch.")
        network = policy_value_network_alpha(config)

        start_epoch = 0
        network.compile(optimizer="adam", loss={
                        "policy": "categorical_crossentropy", "value": "binary_crossentropy"})
        models.save_model(network, model_path, save_format="tf")

    trainGenerator = AlphaZeroGenerator(replay_buffer, config)
elif method == "mu":
    if not hasattr(config, "mu"):
        print("Mu is not supported for this game.")
        exit(-1)

    models_path = "./data/{}/models/".format(base_dir)
    if os.path.exists(models_path+"pv") and not(args.new):
        print("| Loaded previous instance of the models.")
        pv = models.load_model(models_path+"pv", compile=False)
        state = models.load_model(models_path+"state", compile=False)
        dynamics = models.load_model(models_path+"dyn", compile=False)
        try:
            start_epoch = np.load(models_path+"epoch.npy")
        except:
            start_epoch = 0

        print("Epoch: {}".format(start_epoch))
    else:
        print("| Starting model from scratch.")
        state = representation_network(config)
        print("state")
        pv = prediction_network_mu(config)
        print("policy")
        dynamics = dynamics_network(config)
        print("dynamics")

        start_epoch = 0
        models.save_model(pv, models_path+"pv", save_format="tf")
        models.save_model(dynamics, models_path+"dyn", save_format="tf")
        models.save_model(state, models_path+"state", save_format="tf")

    network = unroll_networks(config, state, pv, dynamics)
    network.compile(optimizer="adam", loss={
        "policy": mu_loss_unrolled_cce(config), "value": mu_loss_unrolled_cce(config), "reward": mu_loss_unrolled_cce(config)})

    trainGenerator = MuGenerator(replay_buffer, config)
else:
    print("Unknown method..")
    exit(-1)

trainDataset = trainGenerator.dataset()

trainDataset = tf.data.Dataset.range(4).interleave(lambda x: trainDataset, num_parallel_calls=4)\
    .prefetch(tf.data.experimental.AUTOTUNE)


# waiting until several bootstrap games have been created.
buffer_thr.preload(config.training.batch)
buffer_thr.start()

# Logs for tensorboard
logdir = "./data/{}/logs/{}/".format(base_dir,
                                     datetime.now().strftime("%Y%m%d-%H%M%S"))
file_writer = tf.summary.create_file_writer(logdir)
file_writer.set_as_default()


class StatsLogger(tf.keras.callbacks.Callback):
    def __init__(self):
        self.n = 0

    def on_epoch_end(self, epoch, logs=None):
        global replay_buffer

        self.n += config.training.epoch
        tf.summary.scalar('generated_states',
                          data=replay_buffer.states_count, step=epoch)
        tf.summary.scalar('Games length', data=sum(
            [len(g.turn) for g in replay_buffer.games[:replay_buffer.max_index]])/replay_buffer.max_index, step=epoch)

        if self.n > config.training.checkpoint_freq and method == "mu":
            self.n = 0
            print("Saving models..")
            np.save(models_path+"epoch.npy", epoch)
            models.save_model(pv, models_path+"pv",
                              save_format="tf", include_optimizer=False)
            models.save_model(dynamics, models_path+"dyn",
                              save_format="tf", include_optimizer=False)
            models.save_model(state, models_path+"state",
                              save_format="tf", include_optimizer=False)

# Callbacks

tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

stats_callback = StatsLogger()

tqdm_callback = tfa.callbacks.TQDMProgressBar()


callbacks = [tqdm_callback, tensorboard_callback, stats_callback]
if method == "alpha":
    # CHECKPOINT
    checkpoint_callback = ModelCheckpoint(
        model_path, verbose=1, save_weights_only=False, save_freq=config.training.checkpoint_freq)

    callbacks.append(checkpoint_callback)

# Main loop.
network.fit(trainDataset, epochs=config.training.n_epoch, steps_per_epoch=config.training.epoch//config.training.batch,
            verbose=0, callbacks=callbacks, initial_epoch=start_epoch)


buffer_thr.stop()
buffer_thr.join()
