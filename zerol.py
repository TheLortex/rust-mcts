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

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

## GAME SETTINGS, make sure this is coherent with the generator and evaluator
game = "breakthrough"
K = 5
HISTORY_LENGTH = 2
INPUT_SHAPE = (HISTORY_LENGTH, K, K, 3)
ACTION_SHAPE = (K, K, 3)

WEIGHT_DECAY = 1e-4

def build_network():
    input   = keras.Input(shape=INPUT_SHAPE, name='board')
    x       = layers.Reshape((K, K, HISTORY_LENGTH*3))(input)
    x       = layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(WEIGHT_DECAY), bias_regularizer=l2(WEIGHT_DECAY))(x)
    x       = layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(WEIGHT_DECAY), bias_regularizer=l2(WEIGHT_DECAY))(x)
    x       = layers.Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(WEIGHT_DECAY), bias_regularizer=l2(WEIGHT_DECAY))(x)
    #x       = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    #x       = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    #x       = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    #policy  = layers.Dense(ACTION_SHAPE, activation='softmax', name='policy')(x)
    policy  = layers.Conv2D(3, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(WEIGHT_DECAY), bias_regularizer=l2(WEIGHT_DECAY))(x)
    policy  = layers.Flatten()(policy)
    policy  = layers.Activation(activation='softmax')(policy)
    policy  = layers.Reshape(ACTION_SHAPE, name='policy')(policy)

    value   = layers.Flatten()(x)
    value   = layers.Dense((1), activation='sigmoid', name='value', kernel_regularizer=l2(WEIGHT_DECAY), bias_regularizer=l2(WEIGHT_DECAY))(value)

    return keras.Model(inputs=input, outputs={"policy": policy, "value": value})

REPLAY_BUFFER_SIZE        = 100000 # SAVE THE LAST 100k STATES
EPOCH_SIZE                = REPLAY_BUFFER_SIZE
BATCH_SIZE                = 1024
N_EPOCH                   = 50000

SAVE_REPLAY_BUFFER_FREQ   = 2000            # backup replay buffer every _ moves
CHECKPOINT_FREQ           = 5*EPOCH_SIZE   # save model
EVALUATION_FREQ           = 5*EPOCH_SIZE    # evaluate model

# BREAKTHROUGH SETTINGS

training_data_path = "./training_data/{}/".format(game)
if os.path.exists(training_data_path+"/input_data.npy"):
    print("| Loaded replay buffer")
    input_data = np.load(training_data_path+"input_data.npy")
    policy     = np.load(training_data_path+"policy.npy")
    value      = np.load(training_data_path+"value.npy")
    states_count, replay_buffer_index = np.load(training_data_path+"status.npy")
    print("Status: {} / {}".format(states_count, replay_buffer_index))
else:
    print("| Starting replay buffer from scratch")
    os.makedirs(training_data_path, exist_ok=True)
    input_data = np.zeros((REPLAY_BUFFER_SIZE,) + INPUT_SHAPE)
    policy     = np.zeros((REPLAY_BUFFER_SIZE,) + ACTION_SHAPE)
    value      = np.zeros((REPLAY_BUFFER_SIZE, 1))
    states_count, replay_buffer_index = 0, 0

class BufferThread(Thread):
    def __init__(self):
        Thread.__init__(self)
        self.f = None

    def open_fifo(self):
        print("| Waiting for game generator...",end="", flush=True)
        self.f = open("./fifo", mode="rb")
        print("done!")

    def preload(self, limit):
        global replay_buffer_index
        if replay_buffer_index < limit:
            print("| Booting up first games..")
            self.run(limit=limit)
            print("| Done!")

    def run(self, limit=None):
        global input_data, policy, value, states_count, replay_buffer_index

        idx = 0
        self.continuer = True

        if not(limit is None):
            pbar = tqdm(total=limit)
        else:
            pbar = False

        if not self.f:
            self.open_fifo()

        while self.continuer and ((limit is None) or (replay_buffer_index < limit)):
            sz = int.from_bytes(self.f.read(8), byteorder="big")
            #print(sz)
            pickled = self.f.read(sz)
            new_input_data, new_policy, new_value = pickle.loads(pickled)
            input_data[idx] = np.array(new_input_data, dtype=float).reshape(INPUT_SHAPE)
            policy[idx] = np.array(new_policy, dtype=float).reshape(ACTION_SHAPE)
            value[idx] = new_value
            idx += 1
            states_count += 1
            replay_buffer_index = min(replay_buffer_index+1, REPLAY_BUFFER_SIZE)
            if idx == REPLAY_BUFFER_SIZE:
                idx = 0


            if pbar:
                pbar.update(1)

            if idx % SAVE_REPLAY_BUFFER_FREQ   == 0:
                #print("Saving in training_data/")
                np.save(training_data_path+"input_data",input_data)
                np.save(training_data_path+"policy",policy)
                np.save(training_data_path+"value",value)
                np.save(training_data_path+"status",(states_count, replay_buffer_index))
        

        if pbar:
            pbar.close()
    
    def stop(self):
        self.continuer = False


class ZerolGenerator(Sequence):
    def __init__(self, input_data, policy, value):
        self.input_data = input_data
        self.policy     = policy
        self.value      = value
    
    def on_epoch_end(self):
        pass

    def __len__(self):
        return int(np.floor(REPLAY_BUFFER_SIZE / BATCH_SIZE))

    def __getitem__(self, index):
        # only select games that have been generated (not zeros)
        start_batch = index % (replay_buffer_index // BATCH_SIZE)
        begin_idx = start_batch*BATCH_SIZE
        end_idx   = (start_batch+1)*BATCH_SIZE 

        X = self.input_data[begin_idx:end_idx]
        y = {"policy": self.policy[begin_idx:end_idx], "value": self.value[begin_idx:end_idx]}
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
    network = build_network()
    start_epoch = 0
    network.compile(optimizer="adam", loss={"policy": "categorical_crossentropy", "value": "mse"})
    models.save_model(network, model_path, save_format="tf")


buffer_thr = BufferThread()
buffer_thr.preload(BATCH_SIZE) # waiting until several bootstrap games have been created.
buffer_thr.start()

trainGenerator = ZerolGenerator(input_data, policy, value)
# CHECKPOINT
checkpoint_callback = ModelCheckpoint(model_path, verbose=1, save_weights_only=False, save_freq=CHECKPOINT_FREQ)
# LOGS
logdir = "logs/{}".format(game)
file_writer = tf.summary.create_file_writer(logdir)
file_writer.set_as_default()

from subprocess import PIPE, DEVNULL
class StatsLogger(tf.keras.callbacks.Callback):
    def __init__(self):
        self.n = 0

    def on_epoch_end(self, epoch, logs=None):
        self.n += EPOCH_SIZE
        tf.summary.scalar('generated_states', data=states_count, step=epoch)
        if self.n > EVALUATION_FREQ and False:
            self.n = 0
            print("||Evaluating network")
            np.save(model_path+"epoch",epoch)

            perf_against_random  = int(subprocess.run(["zerol_evaluate", "-p", "rand", "--only-result"], stdout=PIPE, stderr=DEVNULL).stdout.splitlines()[0])
            perf_against_flatmc  = int(subprocess.run(["zerol_evaluate", "-p", "flat", "--only-result"], stdout=PIPE, stderr=DEVNULL).stdout.splitlines()[0])
            #perf_against_flat_uct  = subprocess.run(["zerol_evaluate", "-p", "flat_ucb", "--only-result"]).stdout
            perf_against_uct     = int(subprocess.run(["zerol_evaluate", "-p", "uct", "--only-result"], stdout=PIPE, stderr=DEVNULL).stdout.splitlines()[0])
            #perf_against_rave    = subprocess.run(["zerol_evaluate", "-p", "rave", "--only-result"]).stdout
            print("VS RANDOM: {} | VS FLAT: {} | VS UCT: {}".format(perf_against_random, perf_against_flatmc, perf_against_uct))
            tf.summary.scalar('vs_random', data=perf_against_random, step=epoch)
            tf.summary.scalar('vs_flatmc', data=perf_against_flatmc, step=epoch)
            #tf.summary.scalar('vs_flat_ucb', data=perf_against_flat_uct, step=epoch)
            tf.summary.scalar('vs_uct', data=perf_against_uct,step=epoch)
            #tf.summary.scalar('vs_rave', data=perf_against_rave, step=epoch)
            print("||Done!")

tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

stats_callback = StatsLogger()

tqdm_callback = tfa.callbacks.TQDMProgressBar()

network.fit(trainGenerator, epochs=N_EPOCH, verbose=0, callbacks=[tqdm_callback, checkpoint_callback, tensorboard_callback, stats_callback], initial_epoch=start_epoch)


buffer_thr.stop()
buffer_thr.join()
