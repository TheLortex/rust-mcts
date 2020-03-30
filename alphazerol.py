from subprocess import PIPE, DEVNULL
from tensorflow.keras.utils import Sequence
from tqdm import tqdm
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.regularizers import l2
import tensorflow_addons as tfa
import numpy as np
import pickle
from threading import Thread, RLock
import os
import subprocess

from settings import *
from networks import policy_value_network_alpha

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

## GAME SETTINGS, make sure this is coherent with the generator and evaluator
game = "alpha-breakthrough"

# BREAKTHROUGH SETTINGS

training_data_path = "./training_data/{}/".format(game)
if os.path.exists(training_data_path+"replay_buffer.pkl"):
    print("| Loaded replay buffer")
    f = open(training_data_path+"replay_buffer.pkl", "rb")
    replay_buffer = pickle.load(f)
    f.close()
    print("Status: {} / {}".format(replay_buffer.states_count, replay_buffer.max_index))
else:
    print("| Starting replay buffer from scratch")
    os.makedirs(training_data_path, exist_ok=True)
    replay_buffer = ReplayBuffer(0,0,0,[None]*REPLAY_BUFFER_SIZE)


class AlphaZeroGenerator(Sequence):
    def __init__(self, replay_buffer):
        self.replay_buffer = replay_buffer

    def on_epoch_end(self):
        pass

    def __len__(self):
        return int(np.floor(EPOCH_SIZE / BATCH_SIZE))

    def generate_target(self):
        game_id = np.random.randint(self.replay_buffer.max_index)
        game    = self.replay_buffer.games[game_id]

        game_length = len(game.state)
        move_id = np.random.randint(game_length)

        value = 1 if game.turn[move_id] == game.turn[-1] else 0

        return game.state[move_id], game.policy[move_id], value

    def __getitem__(self, index):
        state  = np.zeros((BATCH_SIZE,)+BOARD_SHAPE)
        policy = np.zeros((BATCH_SIZE,)+ACTION_SHAPE)
        value  = np.zeros((BATCH_SIZE,SUPPORT_SHAPE))

        # only select games that have been generated (not zeros)
        start_batch = index % (replay_buffer.index // BATCH_SIZE)
        begin_idx = start_batch*BATCH_SIZE
        end_idx = (start_batch+1)*BATCH_SIZE

        for i in range(BATCH_SIZE):
            state[i], policy[i], value[i] = self.generate_target()

        X = state
        y = {"policy": policy,
             "value": value}
        return X, y


model_path = "models/{}/".format(game)
if os.path.exists(model_path):
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
    network = policy_value_network_alpha()

    start_epoch = 0
    network.compile(optimizer="adam", loss={
                    "policy": "categorical_crossentropy", "value": "binary_crossentropy"})
    models.save_model(network, model_path, save_format="tf")


buffer_thr = BufferThread(replay_buffer, training_data_path)
# waiting until several bootstrap games have been created.
buffer_thr.preload(BATCH_SIZE)
buffer_thr.start()

trainGenerator = AlphaZeroGenerator(replay_buffer)
# CHECKPOINT
checkpoint_callback = ModelCheckpoint(
    model_path, verbose=1, save_weights_only=False, save_freq=CHECKPOINT_FREQ)
# LOGS
logdir = "logs/{}".format(game)
file_writer = tf.summary.create_file_writer(logdir)
file_writer.set_as_default()


class StatsLogger(tf.keras.callbacks.Callback):
    def __init__(self):
        self.n = 0

    def on_epoch_end(self, epoch, logs=None):
        global replay_buffer

        self.n += EPOCH_SIZE
        tf.summary.scalar('generated_states', data=replay_buffer.states_count, step=epoch)



tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

stats_callback = StatsLogger()

tqdm_callback = tfa.callbacks.TQDMProgressBar()

network.fit(trainGenerator, epochs=N_EPOCH, verbose=0, callbacks=[
            tqdm_callback, checkpoint_callback, tensorboard_callback, stats_callback], initial_epoch=start_epoch)


buffer_thr.stop()
buffer_thr.join()
