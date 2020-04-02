
## GAME SETTINGS, make sure this is coherent with the generator and evaluator

BT_K = 5
HISTORY_LENGTH = 2
BOARD_SHAPE    = (HISTORY_LENGTH, BT_K, BT_K, 3)
ACTION_PLANES  = 3
ACTION_SHAPE   = (BT_K, BT_K, ACTION_PLANES)
HIDDEN_PLANES  = 16
HIDDEN_SHAPE   = (BT_K, BT_K, HIDDEN_PLANES)
SUPPORT_SIZE   =  1
SUPPORT_SHAPE  = 2*SUPPORT_SIZE+1

# MUZERO SPECIFIC
N_UNROLL_STEPS = 2
N_TD_STEPS     = 50
DISCOUNT       = 0.997

WEIGHT_DECAY = 1e-4

REPLAY_BUFFER_SIZE        = 5000 # SAVE THE LAST 5k GAMES
EPOCH_SIZE                = 5*REPLAY_BUFFER_SIZE
BATCH_SIZE                = 512
N_EPOCH                   = 50000

SAVE_REPLAY_BUFFER_FREQ   = 64            # backup replay buffer every _ games
CHECKPOINT_FREQ           = 10*EPOCH_SIZE   # save model
EVALUATION_FREQ           = 5*EPOCH_SIZE    # evaluate model

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

from threading import Thread, RLock
import numpy as np
import pickle
from tqdm import tqdm

class BufferThread(Thread):
    def __init__(self, replay_buffer, training_data_path):
        Thread.__init__(self)
        self.f = None
        self.replay_buffer = replay_buffer
        self.training_data_path = training_data_path

    def open_fifo(self):
        print("| Waiting for game generator...", end="", flush=True)
        self.f = open("./fifo", mode="rb")
        print("done!")

    def preload(self, limit):
        if self.replay_buffer.index < limit:
            print("| Booting up first games..")
            self.run(limit=limit)
            print("| Done!")

    def run(self, limit=None):
        self.continuer = True

        if not(limit is None):
            pbar = tqdm(total=limit)
        else:
            pbar = False

        if not self.f:
            self.open_fifo()

        while self.continuer and ((limit is None) or (self.replay_buffer.index < limit)):
            sz = int.from_bytes(self.f.read(8), byteorder="big")
            # print(sz)
            pickled = self.f.read(sz)
            game = pickle.loads(pickled)
            
            new_state  = np.array(game["state"], dtype=float).reshape((-1,)+BOARD_SHAPE)
            new_policy = np.array(game["policy"], dtype=float).reshape((-1,)+ACTION_SHAPE)
            new_value  = np.array(game["value"], dtype=float).reshape((-1))
            new_action = np.array(game["action"], dtype=float).reshape((-1,)+ACTION_SHAPE)
            new_reward = np.array(game["reward"], dtype=float).reshape((-1,))
            
            self.replay_buffer.games[self.replay_buffer.index] = GameEntry(new_state, new_policy, new_value, new_action, new_reward, game["turn"])
            self.replay_buffer.states_count += 1
            self.replay_buffer.max_index = min(self.replay_buffer.max_index + 1, REPLAY_BUFFER_SIZE)
            
            self.replay_buffer.index += 1
            if self.replay_buffer.index == REPLAY_BUFFER_SIZE:
                self.replay_buffer.index = 0

            if pbar:
                pbar.update(1)

            if self.replay_buffer.index % SAVE_REPLAY_BUFFER_FREQ == 0:
                #print("Saving in training_data/")
                f = open(self.training_data_path+"replay_buffer.pkl", "wb")
                pickle.dump(self.replay_buffer, f)
                f.close()

        if pbar:
            pbar.close()

    def stop(self):
        self.continuer = False
