from subprocess import PIPE, DEVNULL
from tensorflow.keras.utils import Sequence
from tqdm import tqdm
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import losses
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.regularizers import l2
import tensorflow_addons as tfa
from tensorflow.python import debug as tf_debug
import numpy as np
import pickle
from threading import Thread, Lock
import os
import subprocess
from collections import namedtuple

from settings import *
from networks import representation_network, dynamics_network, prediction_network_mu, unroll_networks, policy_value_network_alpha

#tf.debugging.enable_check_numerics(
#    stack_height_limit=30, path_length_limit=50
#)


physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


#ReplayBuffer = namedtuple("ReplayBuffer", ["states_count", "index", "games"])
#GameEntry    = namedtuple("GameEntry", ["state", "policy", "value", "action", "reward", "turn"])
class GameEntry:
    def __init__(self, state, policy, value, action, reward, turn):
        super().__init__()
        self.state = state
        self.policy = policy
        self.value = value 
        self.action = action
        self.reward = reward
        self.turn = turn

class ReplayBuffer:
    def __init__(self, states_count, max_index, index, games):
        super().__init__()
        self.states_count = states_count
        self.max_index = max_index
        self.index = index
        self.games = games


training_data_path = "./training_data/{}/".format(game)
if os.path.exists(training_data_path+"/replay_buffer.pkl"):
    print("| Loaded replay buffer")
    f = open(training_data_path+"replay_buffer.pkl", "rb")
    replay_buffer = pickle.load(f)
    f.close()
    print("Status: {} / {}".format(replay_buffer.states_count, replay_buffer.max_index))
else:
    print("| Starting replay buffer from scratch")
    os.makedirs(training_data_path, exist_ok=True)

    replay_buffer = ReplayBuffer(0,0,0,[None]*REPLAY_BUFFER_SIZE)



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
            game = pickle.loads(pickled)
            
            new_state  = np.array(game["state"], dtype=float).reshape((-1,)+BOARD_SHAPE)
            new_policy = np.array(game["policy"], dtype=float).reshape((-1,)+ACTION_SHAPE)
            new_value  = np.array(game["value"], dtype=float).reshape((-1))
            new_action = np.array(game["action"], dtype=float).reshape((-1,)+ACTION_SHAPE)
            new_reward = np.array(game["reward"], dtype=float).reshape((-1,))
            
            replay_buffer.games[replay_buffer.index] = GameEntry(new_state, new_policy, new_value, new_action, new_reward, game["turn"])
            replay_buffer.states_count += 1
            replay_buffer.max_index = min(replay_buffer.max_index + 1, REPLAY_BUFFER_SIZE)
            
            replay_buffer.index += 1
            if replay_buffer.index == REPLAY_BUFFER_SIZE:
                replay_buffer.index = 0

            if pbar:
                pbar.update(1)

            if replay_buffer.index % SAVE_REPLAY_BUFFER_FREQ == 0:
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
    clamped = np.clip(scaled, -SUPPORT_SIZE, SUPPORT_SIZE)

    v1 = np.floor(clamped)
    p1 = clamped - v1
    v2 = v1 + 1
    p2 = 1 - p1

    result = np.zeros(shape=(SUPPORT_SHAPE,))
    result[int(v1) + SUPPORT_SIZE] = p1
    result[int(v2) + SUPPORT_SIZE] = p2
    return result

class ZerolGenerator(Sequence):
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

        target_policy = np.zeros((N_UNROLL_STEPS,)+ACTION_SHAPE)
        target_value  = np.zeros((N_UNROLL_STEPS,)+(SUPPORT_SHAPE,))
        target_reward = np.zeros((N_UNROLL_STEPS,)+(SUPPORT_SHAPE,))
        target_state  = np.zeros((BOARD_SHAPE))
        target_actions= np.zeros((N_UNROLL_STEPS,)+ACTION_SHAPE)

        target_state[:]  = game.state[move_id]

        for t_idx, i in enumerate(range(move_id, move_id + N_UNROLL_STEPS)):

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
                target_reward[t_idx]  = value_to_support(game.reward[i])
                target_value[t_idx]   = value_to_support(value)
                target_actions[t_idx] = game.action[i]
                target_policy[t_idx]  = game.policy[i]
            # game has finished
            else:
                target_reward[t_idx]  = value_to_support(0)
                target_value[t_idx]   = value_to_support(0)
                random_action = (np.random.random(size=len(ACTION_SHAPE)) * ACTION_SHAPE).astype(int)
                target_actions[t_idx][random_action] = 1
                target_policy[t_idx]  = 1/target_policy[t_idx].size # uniform policy.
    
        return target_policy, target_value, target_reward, target_state, target_actions
        

    def __getitem__(self, index):
        policy = np.zeros((BATCH_SIZE,N_UNROLL_STEPS)+ACTION_SHAPE)
        value  = np.zeros((BATCH_SIZE,N_UNROLL_STEPS)+(SUPPORT_SHAPE,))
        reward = np.zeros((BATCH_SIZE,N_UNROLL_STEPS)+(SUPPORT_SHAPE,))
        state  = np.zeros((BATCH_SIZE,)+BOARD_SHAPE)
        actions= np.zeros((BATCH_SIZE,N_UNROLL_STEPS)+ACTION_SHAPE)

        for i in range(BATCH_SIZE):
            res = self.generate_target()
            #policy[i], value[i], reward[i], state[i], actions[i] = res
            policy[i], value[i], reward[i], state[i], actions[i] = res


        X = [actions, state]
        y = {"policy": policy,
             "value":  value,
             "reward": reward}

        #print(np.sum(y["policy"]), np.sum(y["value"]), np.sum(y["reward"]))
        return X, y


models_path = "models/{}/".format(game)
if os.path.exists(models_path+"pv"): # TODO
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
    state = representation_network()
    print("state")
    pv = prediction_network_mu()
    print("policy")
    dynamics = dynamics_network()
    print("dynamics")

    start_epoch = 0
    models.save_model(pv, models_path+"pv", save_format="tf")
    models.save_model(dynamics, models_path+"dyn", save_format="tf")
    models.save_model(state, models_path+"state", save_format="tf")


def custom_loss_policy(y_true, y_pred):
    policy_loss = 0.
    
    for i in range(N_UNROLL_STEPS):
        policy_loss += losses.categorical_crossentropy(y_true[:,i], y_pred[:,i]) / N_UNROLL_STEPS

    return policy_loss

def custom_loss_value(y_true, y_pred):
    value_loss  = 0.
    
    for i in range(N_UNROLL_STEPS):
        value_loss  += losses.categorical_crossentropy(y_true[:,i], y_pred[:,i])  / N_UNROLL_STEPS

    return value_loss

def custom_loss_reward(y_true, y_pred):
    reward_loss = 0.
    
    for i in range(N_UNROLL_STEPS):
        reward_loss += losses.categorical_crossentropy(y_true[:,i], y_pred[:,i]) / N_UNROLL_STEPS

    return reward_loss



unrolled = unroll_networks(state, pv, dynamics)
adam = optimizers.Adam(lr=0.001)
unrolled.compile(optimizer=adam, loss={
                "policy": custom_loss_policy, "value": custom_loss_value, "reward": custom_loss_reward})

buffer_thr = BufferThread()
# waiting until several bootstrap games have been created.
buffer_thr.preload(SAVE_REPLAY_BUFFER_FREQ)
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
        tf.summary.scalar('Games length', data=sum([len(g.turn) for g in replay_buffer.games[:replay_buffer.index]])/replay_buffer.index, step=epoch)

        if self.n > CHECKPOINT_FREQ:
            self.n = 0
            print("Saving models..")
            np.save(models_path+"epoch.npy", epoch)
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
