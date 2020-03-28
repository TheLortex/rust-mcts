
## GAME SETTINGS, make sure this is coherent with the generator and evaluator
muzero = True
game = "breakthrough"
BT_K = 5
HISTORY_LENGTH = 2
BOARD_SHAPE    = (HISTORY_LENGTH, BT_K, BT_K, 3)
ACTION_PLANES  = 3
ACTION_SHAPE   = (BT_K, BT_K, ACTION_PLANES)
HIDDEN_PLANES  = 16
HIDDEN_SHAPE   = (BT_K, BT_K, HIDDEN_PLANES)
SUPPORT_SIZE   =  1
SUPPORT_SHAPE  = 2*SUPPORT_SIZE+1

N_UNROLL_STEPS = 5
N_TD_STEPS     = 30
DISCOUNT       = 0.997

WEIGHT_DECAY = 1e-4

REPLAY_BUFFER_SIZE        = 100000 # SAVE THE LAST 100k STATES
EPOCH_SIZE                = REPLAY_BUFFER_SIZE
BATCH_SIZE                = 1024
N_EPOCH                   = 50000

SAVE_REPLAY_BUFFER_FREQ   = 2000            # backup replay buffer every _ moves
CHECKPOINT_FREQ           = 5*EPOCH_SIZE   # save model
EVALUATION_FREQ           = 5*EPOCH_SIZE    # evaluate model
