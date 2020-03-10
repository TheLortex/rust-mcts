import libzerol

from tensorflow.keras.utils import Sequence
from tqdm import tqdm
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import numpy as np


GAME_BATCH  = 10
BATCH_SIZE  = 2
N_EPOCH     = 10

# BREAKTHROUGH SETTINGS
K = 8

def build_network():
    input   = keras.Input(shape=(2*K*K+1), name='board')
    x       = layers.Dense((1*K*K), activation='relu')(input)
    #x       = layers.Dense((6*K*K), activation='relu')(x)
    policy  = layers.Dense((3*K*K), activation='softmax', name='policy')(x)
    value   = layers.Dense((1), activation='sigmoid', name='value')(x)
    return keras.Model(inputs=input, outputs=[policy, value])

network     = build_network()


class ZerolGenerator(Sequence):
    def __init__(self, network):
        self.network = network
        self.on_epoch_end()

    def on_epoch_end(self):
        def callback(board):
            res = self.network(board)
            return (res[0].numpy(), res[1].numpy().item())
        print("Calling PUCT")
        self.input_data, self.policy, self.value = libzerol.puct(GAME_BATCH, callback)
        print("PUCT has finished")

    def __len__(self):
        return int(np.floor(GAME_BATCH / BATCH_SIZE))

    def __getitem__(self, index):
        X = self.input_data[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
        y = {"policy": self.policy[index*BATCH_SIZE:(index+1)*BATCH_SIZE], "value": self.value[index*BATCH_SIZE:(index+1)*BATCH_SIZE]}
        return X, y

network.compile(optimizer="adam", loss={"policy": "categorical_crossentropy", "value": "binary_crossentropy"})

trainGenerator = ZerolGenerator(network)
network.fit_generator(trainGenerator, epochs=N_EPOCH, verbose=1)