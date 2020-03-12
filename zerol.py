from tensorflow.keras.utils import Sequence
from tqdm import tqdm
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.callbacks import ModelCheckpoint
from keras_tqdm import TQDMCallback
import numpy as np
import pickle
from threading import Thread, RLock
import os
import subprocess

def build_network():
    input   = keras.Input(shape=(2*K*K+1), name='board')
    x       = layers.Dense((3*K*K), activation='relu')(input)
    x       = layers.Dense((6*K*K), activation='relu')(x)
    x       = layers.Dense((3*K*K), activation='relu')(x)
    policy  = layers.Dense((3*K*K), activation='softmax', name='policy')(x)
    value   = layers.Dense((1), activation='sigmoid', name='value')(x)
    return keras.Model(inputs=input, outputs={"policy": policy, "value": value})

REPLAY_BUFFER_SIZE = 1280000 # SAVE THE LAST 12800 STATES
SAVE_REPLAY_BUFFER_PERIOD = 4096 # backup replay buffer once in a while
BATCH_SIZE         = 16000
N_EPOCH            = 10000


# BREAKTHROUGH SETTINGS
game = "breakthrough"
K = 5

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
    input_data = np.zeros((REPLAY_BUFFER_SIZE, 2*K*K+1))
    policy     = np.zeros((REPLAY_BUFFER_SIZE, 3*K*K))
    value      = np.zeros((REPLAY_BUFFER_SIZE, 1))
    states_count, replay_buffer_index = 0, 0

class BufferThread(Thread):
    def __init__(self):
        Thread.__init__(self)
        self.f = open("./fifo", mode="rb")

    def preload(self, limit):
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

        while self.continuer and ((limit is None) or (replay_buffer_index < limit)):
            sz = int.from_bytes(self.f.read(8), byteorder="big")
            #print(sz)
            pickled = self.f.read(sz)
            new_input_data, new_policy, new_value = pickle.loads(pickled)
            input_data[idx] = new_input_data
            policy[idx] = new_policy
            value[idx] = new_value
            idx += 1
            states_count += 1
            replay_buffer_index = min(replay_buffer_index+1, REPLAY_BUFFER_SIZE)
            if idx == REPLAY_BUFFER_SIZE:
                idx = 0


            if pbar:
                pbar.update(1)

            if idx % SAVE_REPLAY_BUFFER_PERIOD == 0:
                print("Saving in training_data/")
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
    network.compile(optimizer="adam", loss={"policy": "categorical_crossentropy", "value": "binary_crossentropy"})
    models.save_model(network, model_path, save_format="tf")


buffer_thr = BufferThread()
buffer_thr.preload(BATCH_SIZE) # waiting until several bootstrap games have been created.
buffer_thr.start()

trainGenerator = ZerolGenerator(input_data, policy, value)
# CHECKPOINT
CHECKPOINT_PERIOD = 10
checkpoint_callback = ModelCheckpoint(model_path, verbose=1, save_weights_only=False, period=CHECKPOINT_PERIOD)
# LOGS
EVALUATION_PERIOD = 100
logdir = "logs/{}".format(game)
file_writer = tf.summary.create_file_writer(logdir)
file_writer.set_as_default()

from subprocess import PIPE, DEVNULL
class StatsLogger(tf.keras.callbacks.Callback):
    def __init__(self):
        pass

    def on_epoch_end(self, epoch, logs=None):
        tf.summary.scalar('generated_states', data=states_count, step=epoch)
        if epoch % EVALUATION_PERIOD == EVALUATION_PERIOD-1:
            print("||Evaluating network")
            np.save(model_path+"epoch",epoch)

            perf_against_random  = int(subprocess.run(["zerol_evaluate", "-p", "rand", "--only-result"], stdout=PIPE, stderr=DEVNULL).stdout.splitlines()[0])
            #perf_against_flatmc  = subprocess.run(["zerol_evaluate", "-p", "flat", "--only-result"]).stdout
            #perf_against_flat_uct  = subprocess.run(["zerol_evaluate", "-p", "flat_ucb", "--only-result"]).stdout
            perf_against_uct     = int(subprocess.run(["zerol_evaluate", "-p", "uct", "--only-result"], stdout=PIPE, stderr=DEVNULL).stdout.splitlines()[0])
            #perf_against_rave    = subprocess.run(["zerol_evaluate", "-p", "rave", "--only-result"]).stdout
            print("VS RANDOM: {} | VS UCT: {}".format(perf_against_random, perf_against_uct))
            tf.summary.scalar('vs_random', data=perf_against_random, step=epoch)
            #tf.summary.scalar('vs_flatmc', data=perf_against_flatmc, step=epoch)
            #tf.summary.scalar('vs_flat_ucb', data=perf_against_flat_uct, step=epoch)
            tf.summary.scalar('vs_uct', data=perf_against_uct,step=epoch)
            #tf.summary.scalar('vs_rave', data=perf_against_rave, step=epoch)
            print("||Done!")
        
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

stats_callback = StatsLogger()

network.fit(trainGenerator, epochs=N_EPOCH, verbose=0, callbacks=[TQDMCallback(), checkpoint_callback, tensorboard_callback, stats_callback], initial_epoch=start_epoch)

buffer_thr.stop()
buffer_thr.join()
