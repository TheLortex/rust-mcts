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
from collections import namedtuple

from settings import *
from networks import representation_network, dynamics_network, prediction_network_mu, unroll_networks, policy_value_network_alpha

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


ReplayBuffer = namedtuple("ReplayBuffer", ["states_count", "index", "games"])
GameEntry    = namedtuple("GameEntry", ["state", "policy", "value", "action", "reward", "turn"])

training_data_path = "./training_data/{}/".format(game)
if os.path.exists(training_data_path+"/replay_buffer.npy"):
    print("| Loaded replay buffer")
    f = open(training_data_path+"replay_buffer.pkl", "rb")
    replay_buffer = pickle.load(f)
    f.close()
    print("Status: {} / {}".format(replay_buffer.states_count, replay_buffer.index))
else:
    print("| Starting replay buffer from scratch")
    os.makedirs(training_data_path, exist_ok=True)

    replay_buffer = ReplayBuffer(0,0,[None]*REPLAY_BUFFER_SIZE)


class BufferThread(Thread):
    def __init__(self):
        Thread.__init__(self)
        self.f = None

    def open_fifo(self):
        print("| Waiting for game generator...", end="", flush=True)
        self.f = open("./fifo", mode="rb")
        print("done!")

    def preload(self, limit):
        global replay_buffer
        if replay_buffer.index < limit:
            print("| Booting up first games..")
            self.run(limit=limit)
            print("| Done!")

    def run(self, limit=None):
        global replay_buffer

        idx = 0
        self.continuer = True

        if not(limit is None):
            pbar = tqdm(total=limit)
        else:
            pbar = False

        if not self.f:
            self.open_fifo()

        while self.continuer and ((limit is None) or (replay_buffer.index < limit)):
            sz = int.from_bytes(self.f.read(8), byteorder="big")
            # print(sz)
            pickled = self.f.read(sz)
            new_state, new_policy, new_value, new_action, new_reward = pickle.loads(pickled)
            new_state  = np.array(new_state, dtype=float).reshape((-1,)+BOARD_SHAPE)
            new_policy = np.array(new_policy, dtype=float).reshape((-1,)+ACTION_SHAPE)
            new_value  = np.array(new_value, dtype=float).reshape((-1))
            new_action = np.array(new_action, dtype=float).reshape((-1,)+ACTION_SHAPE)
            new_reward = np.array(new_action, dtype=float).reshape((-1,))
            
            replay_buffer.games[idx] = GameEntry(new_state, new_policy, new_value, new_action, new_reward)
            replay_buffer.states_count += 1
            replay_buffer.index = min(replay_buffer.index + 1, REPLAY_BUFFER_SIZE)

            idx += 1
            if idx == REPLAY_BUFFER_SIZE:
                idx = 0

            if pbar:
                pbar.update(1)

            if idx % SAVE_REPLAY_BUFFER_FREQ == 0:
                #print("Saving in training_data/")
                f = open(training_data_path+"replay_buffer.pkl", "wb")
                pickle.dump(replay_buffer, f)
                f.close()

        if pbar:
            pbar.close()

    def stop(self):
        self.continuer = False

# scalar to categorical transformation.
def value_to_support(v):
    # invertible transformation
    scaled = np.sign(v) * ((np.sqrt(np.abs(v)+1)-1)) + 0.001*v
    # clamp to support
    clamped = np.clamped(scaled, -SUPPORT_SIZE, SUPPORT_SIZE)

    v1 = np.floor(clamped)
    p1 = clamped - v1
    v2 = v1 + 1
    p2 = 1 - p1

    result = np.zeros(shape=SUPPORT_SHAPE)
    result[v1 + SUPPORT_SIZE + 1] = p1
    result[v2 + SUPPORT_SIZE + 1] = p2
    return result

class ZerolGenerator(Sequence):
    def __init__(self, replay_buffer):
        self.replay_buffer = replay_buffer

    def on_epoch_end(self):
        pass

    def __len__(self):
        return int(np.floor(REPLAY_BUFFER_SIZE / BATCH_SIZE))

    def generate_target(self):
        game_id = np.random.randint(self.replay_buffer.index)
        game    = self.replay_buffer.games[game_id]
        game_length = len(game.state)
        move_id = np.random.randint(game_length)

        policy = np.zeros((N_UNROLL_STEPS)+ACTION_SHAPE)
        value  = np.zeros((N_UNROLL_STEPS)+SUPPORT_SHAPE)
        reward = np.zeros((N_UNROLL_STEPS)+SUPPORT_SHAPE)
        state  = np.zeros((BOARD_SHAPE))
        actions= np.zeros((N_UNROLL_STEPS)+ACTION_SHAPE)

        state[:]  = game.state[move_id]

        for i in range(move_id, move_id + N_UNROLL_STEPS):

            # compute target value
            value = 0
            if i+N_TD_STEPS < game_length:
                value += game.value[i + N_TD_STEPS] * DISCOUNT ** N_TD_STEPS

            for j, reward in enumerate(game.reward[i:i+N_TD_STEPS]):
                discounted_reward = reward * DISCOUNT ** j
                if game.turn[i+j] == game.turn[i]:
                    value += discounted_reward
                else:
                    value -= discounted_reward
            
            # still in game
            if i < game_length:
                reward[i]  = value_to_support(game.reward[i])
                value[i]   = value_to_support(value)
                actions[i] = game.action[i]
                policy[i]  = game.policy[i]
            # game has finished
            else:
                reward[i]  = value_to_support(0)
                value[i]   = value_to_support(0)
                random_action = (np.random.random(size=len(ACTION_SHAPE)) * ACTION_SHAPE).astype(int)
                actions[i][random_action] = 1
                policy[i]  = 1/policy[i].size # uniform policy.
        

    def __getitem__(self, index):
        policy = np.zeros((BATCH_SIZE,N_UNROLL_STEPS)+ACTION_SHAPE)
        value  = np.zeros((BATCH_SIZE,N_UNROLL_STEPS)+SUPPORT_SHAPE)
        reward = np.zeros((BATCH_SIZE,N_UNROLL_STEPS)+SUPPORT_SHAPE)
        state  = np.zeros((BATCH_SIZE,BOARD_SHAPE))
        actions= np.zeros((BATCH_SIZE,N_UNROLL_STEPS)+ACTION_SHAPE)

        for i in range(BATCH_SIZE):
            policy[i], value[i], reward[i], state[i], actions[i] = self.generate_target()

        X = {"initial_state": state, "actions": actions}
        y = {"policy": policy,
             "value": value,
             "reward": reward}
        return X, y


models_path = "models/{}/".format(game)
if os.path.exists(models_path+"pv"):
    print("| Loaded previous instance of the models.")
    pv = models.load_model(models_path+"pv")
    dynamics = models.load_model(models_path+"dyn")
    state = models.load_model(models_path+"state")
    
    start_epoch = 0
    print("Epoch: {}".format(start_epoch))
else:
    print("| Starting model from scratch.")
    state = representation_network()
    print("state")
    pv = prediction_network_mu()
    print("policy")
    dynamics = dynamics_network()
    print("dynamics")

    unrolled = unroll_networks(state, pv, dynamics)
    unrolled.compile(optimizer="adam", loss={
                    "policy": "categorical_crossentropy", "value": "categorical_crossentropy", "reward": "categorical_crossentropy"})

    start_epoch = 0
    models.save_model(pv, models_path+"pv", save_format="tf")
    models.save_model(dynamics, models_path+"dyn", save_format="tf")
    models.save_model(state, models_path+"state", save_format="tf")


buffer_thr = BufferThread()
# waiting until several bootstrap games have been created.
buffer_thr.preload(BATCH_SIZE)
buffer_thr.start()

trainGenerator = ZerolGenerator(replay_buffer)
# CHECKPOINT
#checkpoint_callback = ModelCheckpoint(
#    model_path, verbose=1, save_weights_only=False, save_freq=CHECKPOINT_FREQ)
# LOGS
logdir = "logs/{}".format(game)
file_writer = tf.summary.create_file_writer(logdir)
file_writer.set_as_default()


class StatsLogger(tf.keras.callbacks.Callback):
    def __init__(self):
        self.n = 0

    def on_epoch_end(self, epoch, logs=None):
        global pv, dynamics, state
        
        self.n += EPOCH_SIZE
        tf.summary.scalar('Generated games', data=replay_buffer.states_count, step=epoch)
        tf.summary.scalar('Games length', data=sum([len(g['turn']) for g in replay_buffer.games[:replay_buffer.index]])/replay_buffer.index)

        if self.n > CHECKPOINT_FREQ:
            self.n = 0
            print("Saving models..")
            models.save_model(pv, models_path+"pv", save_format="tf")
            models.save_model(dynamics, models_path+"dyn", save_format="tf")
            models.save_model(state, models_path+"state", save_format="tf")


tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

stats_callback = StatsLogger()

tqdm_callback = tfa.callbacks.TQDMProgressBar()

unrolled.fit(trainGenerator, epochs=N_EPOCH, verbose=0, callbacks=[
            tqdm_callback, tensorboard_callback, stats_callback], initial_epoch=start_epoch)


buffer_thr.stop()
buffer_thr.join()
