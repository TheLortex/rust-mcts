from tensorflow.keras.utils import Sequence
from tqdm import tqdm
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import pickle
from threading import Thread, RLock

K = 8

def build_network():
    input   = keras.Input(shape=(2*K*K+1), name='board')
    x       = layers.Dense((1*K*K), activation='relu')(input)
    #x       = layers.Dense((6*K*K), activation='relu')(x)
    policy  = layers.Dense((3*K*K), activation='softmax', name='policy')(x)
    value   = layers.Dense((1), activation='sigmoid', name='value')(x)
    return keras.Model(inputs=input, outputs=[policy, value])

#network = build_network()
#network.compile(optimizer="adam", loss={"policy": "categorical_crossentropy", "value": "binary_crossentropy"})
#models.save_model(network, "models/sample", include_optimizer=False, save_format="tf")

REPLAY_BUFFER = 128000 # SAVE THE LAST 12800 GAMES
BATCH_SIZE    = 128
N_EPOCH       = 1000

SAVE_BUFFER   = True 
SAVE_NETWORK  = True

# BREAKTHROUGH SETTINGS
K = 8


input_data = np.zeros((REPLAY_BUFFER, 2*K*K+1))
policy     = np.zeros((REPLAY_BUFFER, 3*K*K))
value      = np.zeros((REPLAY_BUFFER, 1))

class BufferThread(Thread):
    def __init__(self):
        Thread.__init__(self)

    def run(self):
        f = open("./fifo", mode="rb")
        idx = 0
        self.continuer = True
        while self.continuer:
            sz = int.from_bytes(f.read(8), byteorder="big")
            pickled = f.read(sz)
            new_input_data, new_policy, new_value = pickle.loads(pickled)
            input_data[idx] = new_input_data
            policy[idx] = new_policy
            value[idx] = new_value
            idx += 1
            if idx == REPLAY_BUFFER:
                if SAVE_BUFFER:
                    print("Full buffer cycle! Saving in training_data/")
                    np.save("./training_data/input_data.np",input_data)
                    np.save("./training_data/policy.np",policy)
                    np.save("./training_data/value.np",value)
                idx = 0
    
    def stop(self):
        self.continuer = False


class ZerolGenerator(Sequence):
    def __init__(self, input_data, policy, value):
        self.input_data = input_data
        self.policy     = policy
        self.value      = value
    
    def on_epoch_end(self):
        pass
        #print("Saving model..")
        #models.save_model(self.network, "models/sample", include_optimizer=False, save_format="tf")

    def __len__(self):
        return int(np.floor(REPLAY_BUFFER / BATCH_SIZE))

    def __getitem__(self, index):
        X = self.input_data[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
        y = {"policy": self.policy[index*BATCH_SIZE:(index+1)*BATCH_SIZE], "value": self.value[index*BATCH_SIZE:(index+1)*BATCH_SIZE]}
        return X, y

buffer_thr = BufferThread()
buffer_thr.start()

network = build_network()
network.compile(optimizer="adam", loss={"policy": "categorical_crossentropy", "value": "binary_crossentropy"})
trainGenerator = ZerolGenerator(input_data, policy, value)

checkpoint = ModelCheckpoint("models/sample", verbose=1, save_weights_only=False)

try:
    network.fit_generator(trainGenerator, epochs=N_EPOCH, verbose=1, callbacks=[checkpoint])
except KeyboardInterrupt:
    buffer_thr.stop()
    raise
except:
    raise

buffer_thr.stop()
buffer_thr.join()
