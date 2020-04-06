
def dir_name(config, method):
    if config.game.kind == "Breakthrough":
        return "{}-breakthrough-{}".format(method, config.game.size)
    elif config.game.kind == "Gym":
        return "{}-gym-{}".format(method, config.game.name)
    else:
        print("Unknown game in config file.")
        exit(-1)

def get_board_shape(config):
    if config.game.kind == "Breakthrough":
        return (config.game.history, config.game.size, config.game.size, 3)
    elif config.game.kind == "Gym":
        if config.game.name == "Breakout-v0":
            return (config.game.history, 96, 96, 3)
        else:
            print("Gym not implemented for this game.")
            exit(-1)
    else:
        print("Unknown game in config file.")
        exit(-1)

def get_action_shape(config):
    if config.game.kind == "Breakthrough":
        return (config.game.size, config.game.size, 3)
    elif config.game.kind == "Gym":
        if config.game.name == "Breakout-v0":
            return (4,)
        else:
            print("Gym not implemented for this game.")
            exit(-1)
    else:
        print("Unknown game in config file.")
        exit(-1)

    
import numpy as np
# scalar to categorical transformation.
def value_to_support(v, support_size):
    # invertible transformation
    scaled = np.sign(v) * ((np.sqrt(np.abs(v)+1)-1)) + 0.001*v
    # clamp to support
    clamped = np.clip(scaled, -support_size, support_size)

    v1 = np.floor(clamped)
    p1 = 1 - (clamped - v1)
    v2 = v1 + 1
    p2 = 1 - p1

    result = np.zeros(shape=(support_size*2+1,))
    result[int(v1) + support_size] = p1
    if int(v2) + support_size < support_size*2+1:
        result[int(v2) + support_size] = p2
    return result
                                                                                                                       
from tensorflow.keras import losses
def mu_loss_unrolled_cce(config):
    def loss(y_true, y_pred):
        policy_loss = 0.

        for i in range(config.mu.unroll_steps):
            policy_loss += losses.categorical_crossentropy(
                y_true[:, i], y_pred[:, i]) / config.mu.unroll_steps

        return policy_loss
    return loss

def get_support_shape(x):
    return (x or 0)*2+1
"""
## GAME SETTINGS, make sure this is coherent with the generator and evaluator
GAME = "breakthrough"

if GAME == "breakthrough":
    BT_K = 5
    HISTORY_LENGTH = 2
    BOARD_SHAPE    = (HISTORY_LENGTH, BT_K, BT_K, 3)
    ACTION_PLANES  = 3
    ACTION_SHAPE   = (BT_K, BT_K, ACTION_PLANES)
    HIDDEN_PLANES  = 16
    HIDDEN_SHAPE   = (BT_K, BT_K, HIDDEN_PLANES)
    SUPPORT_SIZE   =  1
elif GAME == "atari":
    HISTORY_LENGTH = 8
    BOARD_SHAPE    = (HISTORY_LENGTH, 96, 96, 3)
    ACTION_PLANES  = 4 # breakout
    ACTION_SHAPE   = (ACTION_PLANES, )
    HIDDEN_PLANES  = 16
    HIDDEN_SHAPE   = (6, 6, HIDDEN_PLANES)
    SUPPORT_SIZE   = 300

SUPPORT_SHAPE  = 2*SUPPORT_SIZE+1


# MUZERO SPECIFIC
N_UNROLL_STEPS = 5
N_TD_STEPS     = 300
DISCOUNT       = 0.997

WEIGHT_DECAY = 1e-4

REPLAY_BUFFER_SIZE        = 5000 # SAVE THE LAST 5k GAMES
EPOCH_SIZE                = 5*REPLAY_BUFFER_SIZE
BATCH_SIZE                = 512
N_EPOCH                   = 50000

SAVE_REPLAY_BUFFER_FREQ   = 64            # backup replay buffer every _ games
CHECKPOINT_FREQ           = 5*EPOCH_SIZE   # save model
EVALUATION_FREQ           = 5*EPOCH_SIZE    # evaluate model
"""