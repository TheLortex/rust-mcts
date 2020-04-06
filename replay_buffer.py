from threading import Thread, RLock
import numpy as np
import pickle
from tqdm import tqdm
from settings import *
import tensorflow as tf
import os
from tensorflow.keras.utils import Sequence

#  /$$$$$$$  /$$$$$$$$ /$$$$$$$  /$$        /$$$$$$  /$$     /$$       /$$$$$$$  /$$   /$$ /$$$$$$$$ /$$$$$$$$ /$$$$$$$$ /$$$$$$$ 
# | $$__  $$| $$_____/| $$__  $$| $$       /$$__  $$|  $$   /$$/      | $$__  $$| $$  | $$| $$_____/| $$_____/| $$_____/| $$__  $$
# | $$  \ $$| $$      | $$  \ $$| $$      | $$  \ $$ \  $$ /$$/       | $$  \ $$| $$  | $$| $$      | $$      | $$      | $$  \ $$
# | $$$$$$$/| $$$$$   | $$$$$$$/| $$      | $$$$$$$$  \  $$$$/        | $$$$$$$ | $$  | $$| $$$$$   | $$$$$   | $$$$$   | $$$$$$$/
# | $$__  $$| $$__/   | $$____/ | $$      | $$__  $$   \  $$/         | $$__  $$| $$  | $$| $$__/   | $$__/   | $$__/   | $$__  $$
# | $$  \ $$| $$      | $$      | $$      | $$  | $$    | $$          | $$  \ $$| $$  | $$| $$      | $$      | $$      | $$  \ $$
# | $$  | $$| $$$$$$$$| $$      | $$$$$$$$| $$  | $$    | $$          | $$$$$$$/|  $$$$$$/| $$      | $$      | $$$$$$$$| $$  | $$
# |__/  |__/|________/|__/      |________/|__/  |__/    |__/          |_______/  \______/ |__/      |__/      |________/|__/  |__/

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

class BufferThread(Thread):
    def __init__(self, config, replay_buffer, training_data_path, fifo_path="./fifo"):
        Thread.__init__(self)
        self.f = None
        self.config = config
        self.replay_buffer = replay_buffer
        self.training_data_path = training_data_path
        self.fifo_path = fifo_path

    def open_fifo(self):
        print("| Waiting for game generator...", end="", flush=True)
        if not(os.path.exists(self.fifo_path)):
            os.mkfifo(self.fifo_path)
        self.f = open(self.fifo_path, mode="rb")
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
            
            action_shape = get_action_shape(self.config)

            new_state  = np.array(game["state"], dtype=float).reshape((-1,)+get_board_shape(self.config))
            new_policy = np.array(game["policy"], dtype=float).reshape((-1,)+action_shape)
            new_value  = np.array(game["value"], dtype=float).reshape((-1))
            new_action = np.array(game["action"], dtype=float).reshape((-1,)+action_shape)
            new_reward = np.array(game["reward"], dtype=float).reshape((-1,))
            
            self.replay_buffer.games[self.replay_buffer.index] = GameEntry(new_state, new_policy, new_value, new_action, new_reward, game["turn"])
            self.replay_buffer.states_count += 1
            self.replay_buffer.max_index = min(self.replay_buffer.max_index + 1, self.config.training.replay_buffer)
            
            self.replay_buffer.index += 1
            if self.replay_buffer.index == self.config.training.replay_buffer:
                self.replay_buffer.index = 0

            if pbar:
                pbar.update(1)

            if self.replay_buffer.index % self.config.training.save_replay_freq == 0:
                #print("Saving in training_data/")
                f = open(self.training_data_path+"replay_buffer.pkl", "wb")
                pickle.dump(self.replay_buffer, f)
                f.close()

        if pbar:
            pbar.close()

    def stop(self):
        self.continuer = False

#  /$$      /$$ /$$   /$$          /$$$$$$  /$$$$$$$$ /$$   /$$
# | $$$    /$$$| $$  | $$         /$$__  $$| $$_____/| $$$ | $$
# | $$$$  /$$$$| $$  | $$        | $$  \__/| $$      | $$$$| $$
# | $$ $$/$$ $$| $$  | $$ /$$$$$$| $$ /$$$$| $$$$$   | $$ $$ $$
# | $$  $$$| $$| $$  | $$|______/| $$|_  $$| $$__/   | $$  $$$$
# | $$\  $ | $$| $$  | $$        | $$  \ $$| $$      | $$\  $$$
# | $$ \/  | $$|  $$$$$$/        |  $$$$$$/| $$$$$$$$| $$ \  $$
# |__/     |__/ \______/          \______/ |________/|__/  \__/
#                                                                                                                         


class MuGenerator(Sequence):
    def __init__(self, replay_buffer, config):
        self.replay_buffer = replay_buffer
        self.config = config

    def on_epoch_end(self):
        pass

    def __len__(self):
        return int(np.floor(self.config.epoch / self.config.batch))

    def generate_target(self):

        game_id = np.random.randint(self.replay_buffer.max_index)
        game = self.replay_buffer.games[game_id]

        game_length = len(game.state)
        move_id = np.random.randint(game_length)

        action_shape = get_action_shape(self.config)


        target_policy = np.zeros((self.config.mu.unroll_steps,)+action_shape)
        target_value = np.zeros((self.config.mu.unroll_steps,self.config.mu.puct.value_support*2+1))
        target_reward = np.zeros((self.config.mu.unroll_steps,self.config.mu.reward_support*2+1))
        target_state = np.zeros(get_board_shape(self.config))
        target_actions = np.zeros((self.config.mu.unroll_steps,)+action_shape)

        target_state[:] = game.state[move_id]

        for t_idx, i in enumerate(range(move_id, move_id + self.config.mu.unroll_steps)):

            # compute target value
            value = 0
            if i+self.config.mu.td_steps < game_length:
                value += game.value[i + self.config.mu.td_steps] * self.config.mu.puct.discount ** self.config.mu.td_steps

            for j, reward in enumerate(game.reward[i:i+self.config.mu.td_steps]):
                discounted_reward = reward * self.config.mu.puct.discount ** j
                if game.turn[i+j] == game.turn[i]:
                    value += discounted_reward
                else:
                    value -= discounted_reward

            # still in game
            if i < game_length:
                target_reward[t_idx] = value_to_support(game.reward[i], self.config.mu.reward_support)
                target_value[t_idx] = value_to_support(value, self.config.mu.puct.value_support)
                target_actions[t_idx] = game.action[i]
                target_policy[t_idx] = game.policy[i]
            # game has finished
            else:
                target_reward[t_idx] = value_to_support(0, self.config.mu.reward_support)
                target_value[t_idx] = value_to_support(0, self.config.mu.puct.value_support)
                random_action = (np.random.random(
                    size=len(action_shape)) * action_shape).astype(int)
                target_actions[t_idx][random_action] = 1
                # uniform policy.
                target_policy[t_idx] = 1/target_policy[t_idx].size

        return target_policy, target_value, target_reward, target_state, target_actions

    def __getitem__(self, index):        
        action_shape = get_action_shape(self.config)
        board_shape  = get_board_shape(self.config)
        batch_size = self.config.training.batch
        n_unroll_steps = self.config.mu.unroll_steps

        policy = np.zeros((batch_size, n_unroll_steps)+action_shape)
        value = np.zeros((batch_size, n_unroll_steps, get_support_shape(self.config.mu.puct.value_support)))
        reward = np.zeros((batch_size, n_unroll_steps, self.config.mu.reward_support*2+1))
        state = np.zeros((batch_size,)+board_shape)
        actions = np.zeros((batch_size, n_unroll_steps)+action_shape)

        for i in range(batch_size):
            res = self.generate_target()
            policy[i], value[i], reward[i], state[i], actions[i] = res

        X = {"actions": actions, "starting_board": state}
        y = {"policy": policy,
             "value":  value,
             "reward": reward}

        # print(np.sum(y["policy"]), np.sum(y["value"]), np.sum(y["reward"]))
        return X, y

    def generate(self):
        for _ in range(self.config.training.epoch):
            yield self[0]

    def dataset(self):
        action_shape = get_action_shape(self.config)
        board_shape  = get_board_shape(self.config)
        batch_size = self.config.training.batch
        n_unroll_steps = self.config.mu.unroll_steps

        shapes = ({"actions": tf.TensorShape((None, n_unroll_steps,)+action_shape), "starting_board": tf.TensorShape((None,)+ board_shape)}, {
            "reward": tf.TensorShape((None, n_unroll_steps,self.config.mu.reward_support*2+1)),
            "policy": tf.TensorShape((None, n_unroll_steps,)+action_shape),
            "value": tf.TensorShape((None, n_unroll_steps,get_support_shape(self.config.mu.puct.value_support)))
        })
        trainDataset = tf.data.Dataset.from_generator(self.generate,
                                                    output_types=({"actions": tf.float32, "starting_board": tf.float32}, {"policy": tf.float32, "value": tf.float32, "reward": tf.float32}), output_shapes=shapes)
        return trainDataset


#   /$$$$$$  /$$       /$$$$$$$  /$$   /$$  /$$$$$$           /$$$$$$  /$$$$$$$$ /$$   /$$
#  /$$__  $$| $$      | $$__  $$| $$  | $$ /$$__  $$         /$$__  $$| $$_____/| $$$ | $$
# | $$  \ $$| $$      | $$  \ $$| $$  | $$| $$  \ $$        | $$  \__/| $$      | $$$$| $$
# | $$$$$$$$| $$      | $$$$$$$/| $$$$$$$$| $$$$$$$$ /$$$$$$| $$ /$$$$| $$$$$   | $$ $$ $$
# | $$__  $$| $$      | $$____/ | $$__  $$| $$__  $$|______/| $$|_  $$| $$__/   | $$  $$$$
# | $$  | $$| $$      | $$      | $$  | $$| $$  | $$        | $$  \ $$| $$      | $$\  $$$
# | $$  | $$| $$$$$$$$| $$      | $$  | $$| $$  | $$        |  $$$$$$/| $$$$$$$$| $$ \  $$
# |__/  |__/|________/|__/      |__/  |__/|__/  |__/         \______/ |________/|__/  \__/
#                                                                                                                                        
# 

class AlphaZeroGenerator(Sequence):
    def __init__(self, replay_buffer, config):
        self.replay_buffer = replay_buffer
        self.config = config

    def on_epoch_end(self):
        pass

    def __len__(self):
        return int(np.floor(self.config.training.epoch_size / self.config.training.batch))

    def generate_target(self):
        game_id = np.random.randint(self.replay_buffer.max_index)
        game    = self.replay_buffer.games[game_id]

        game_length = len(game.state)
        move_id = np.random.randint(game_length)

        value = 1 if game.turn[move_id] == game.turn[-1] else 0

        return game.state[move_id], game.policy[move_id], value

    def __getitem__(self, index):
        state  = np.zeros((self.config.training.batch,)+get_board_shape(self.config))
        policy = np.zeros((self.config.training.batch,)+get_action_shape(self.config))
        value  = np.zeros((self.config.training.batch, get_support_shape(self.config.alpha.puct.value_support)))

        # only select games that have been generated (not zeros)
        start_batch = index % (self.replay_buffer.max_index // self.config.training.batch)
        begin_idx = start_batch*self.config.training.batch
        end_idx = (start_batch+1)*self.config.training.batch

        for i in range(self.config.training.batch):
            state[i], policy[i], value[i] = self.generate_target()

        X = state
        y = {"policy": policy,
             "value": value}
        return X, y
          
    def generate(self):
        for _ in range(2*self.config.training.epoch):
            yield self[0]
                                                 
    def dataset(self):
        action_shape = get_action_shape(self.config)
        board_shape  = get_board_shape(self.config)
        batch_size = self.config.training.batch

        shapes = (tf.TensorShape((None,)+ board_shape), {
            "policy": tf.TensorShape((None,)+action_shape),
            "value": tf.TensorShape((None,get_support_shape(self.config.alpha.puct.value_support)))
        })
        trainDataset = tf.data.Dataset.from_generator(self.generate,
                                                    output_types=(tf.float32, {"policy": tf.float32, "value": tf.float32}), output_shapes=shapes)
        return trainDataset
