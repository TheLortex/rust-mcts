
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.regularizers import l2 

from settings import *


def residual_block(input, name, size=32, activation='relu', convert=False):
    if convert:
        input = layers.Conv2D(size, 3, padding='same', name="res_conv_trans_{}".format(name),
                        kernel_regularizer=l2(WEIGHT_DECAY))(input)
        input = layers.Activation(activation)(input)
        input = layers.BatchNormalization()(input)
    
    X_skip = input
    x1 = layers.Conv2D(size, 3, padding='same', name="res_conv_A_{}".format(name),
                        kernel_regularizer=l2(WEIGHT_DECAY))(input)
    x1 = layers.Activation(activation)(x1)
    x1 = layers.BatchNormalization()(x1)

    x2 = layers.Conv2D(size, 3, padding='same', name="res_conv_B_{}".format(name),
                        kernel_regularizer=l2(WEIGHT_DECAY))(x1)
    x2 = layers.Activation(activation)(x2)
    x2 = layers.BatchNormalization()(x2)

    return layers.Add()([x2, X_skip])

def policy_value_network_alpha():
    input   = keras.Input(shape=BOARD_SHAPE, name='board')
    x       = layers.Reshape(BOARD_SHAPE)(input) # assert board shape
    x       = layers.Permute((2,3,1,4))(x)
    x       = layers.Reshape((BT_K, BT_K, HISTORY_LENGTH*3))(x)

    x       = residual_block(x, "pv_a", convert=True)
    x       = residual_block(x, "pv_b")
    x       = residual_block(x, "pv_c")
    
    policy  = residual_block(x, "pv_d", size=ACTION_PLANES, convert=True)
    policy  = layers.Flatten()(policy)
    policy  = layers.Activation(activation='softmax')(policy)
    policy  = layers.Reshape(ACTION_SHAPE, name='policy')(policy)

    value   = residual_block(x, "pv_e")
    value   = layers.Flatten()(value)
    value   = layers.Dense((1), activation='sigmoid', name='value', kernel_regularizer=l2(WEIGHT_DECAY), bias_regularizer=l2(WEIGHT_DECAY))(value)

    return keras.Model(inputs=input, outputs={"policy": policy, "value": value})

def prediction_network_mu():
    input   = keras.Input(shape=HIDDEN_SHAPE, name='board')
    x       = layers.Reshape(HIDDEN_SHAPE, name='PredictionNetworkBoard')(input)

    x       = residual_block(x, "pred_a", convert=True)
    x       = residual_block(x, "pred_b")
    x       = residual_block(x, "pred_c")
    policy  = residual_block(x, "pred_d", size=ACTION_PLANES, convert=True)

    policy  = layers.Flatten()(policy)
    policy  = layers.Activation(activation='softmax')(policy)
    policy  = layers.Reshape(ACTION_SHAPE, name='policy')(policy)

    value   = residual_block(x, "pred_e")
    value   = layers.Flatten()(value)
    value   = layers.Dense((2*SUPPORT_SIZE+1), activation='softmax', name='value', kernel_regularizer=l2(WEIGHT_DECAY), bias_regularizer=l2(WEIGHT_DECAY))(value)

    return keras.Model(inputs=input, outputs={"policy": policy, "value": value}, name="Prediction")

def representation_network():
    input   = keras.Input(shape=BOARD_SHAPE, name='board')
    x       = layers.Reshape(BOARD_SHAPE)(input) # assert board shape
    x       = layers.Permute((2,3,1,4))(x)
    x       = layers.Reshape((BT_K, BT_K, HISTORY_LENGTH*3), name='RepresentationNetworkBoard')(x)

    x       = residual_block(x, "repr_a", convert=True)
    x       = residual_block(x, "repr_b")
    x       = residual_block(x, "repr_c")

    repr_board   = layers.Conv2D(HIDDEN_PLANES, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(WEIGHT_DECAY), bias_regularizer=l2(WEIGHT_DECAY), name='repr_board')(x)

    return keras.Model(inputs=input, outputs=repr_board, name="Representation")

def representation_network_atari():
    input   = keras.Input(shape=BOARD_SHAPE, name='board')
    x       = layers.Reshape(BOARD_SHAPE)(input) # assert board shape
    x       = layers.Permute((2,3,1,4))(x)
    x       = layers.Reshape((96,96,HISTORY_LENGTH*3))(x)
    x       = layers.Conv2D(32, 3, padding='same', strides=2)(x)
    x       = residual_block(x, "repr_a", size=32)
    x       = layers.Conv2D(64, 3, padding='same', strides=2)(x)
    x       = residual_block(x, "repr_b", size=64)
    x       = layers.AveragePooling2D()(x)
    x       = residual_block(x, "repr_c", size=64)
    x       = layers.AveragePooling2D()(x)
    x       = residual_block(x, "repr_d", size=64)
    repr_board   = layers.Conv2D(HIDDEN_PLANES, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(WEIGHT_DECAY), bias_regularizer=l2(WEIGHT_DECAY), name='repr_board')(x)

    return keras.Model(inputs=input, outputs=repr_board, name="Representation")


def dynamics_network():
    input_board  = keras.Input(shape=HIDDEN_SHAPE, name='board')
    input_board_  = layers.Reshape(HIDDEN_SHAPE, name='DynamicsNetworkState')(input_board)
    input_action = keras.Input(shape=ACTION_SHAPE, name='action')
    input_action_  = layers.Reshape(ACTION_SHAPE, name='DynamicsNetworkAction')(input_action)
    if GAME == "atari":
        input_action_ = layers.RepeatVector(HIDDEN_SHAPE[0]**2)(input_action_)
        input_action_ = layers.Reshape((HIDDEN_SHAPE[0], HIDDEN_SHAPE[1])+ACTION_SHAPE)(input_action_)
    x            = layers.Concatenate(axis=-1)([input_board_, input_action_])#, (BT_K, BT_K, HIDDEN_PLANES + ACTION_PLANES))

    x       = residual_block(x, "dyn_a", convert=True)
    x       = residual_block(x, "dyn_b")
    x       = residual_block(x, "dyn_c")

    next_board    = residual_block(x, "dyn_d", size=HIDDEN_PLANES, convert=True)
    #max_state    = layers.Reshape((1,1,-1))(layers.Lambda(lambda x: K.max(x, axis=1, keepdims=True))(layers.Reshape((BT_K*BT_K, -1))(next_board)))
    #min_state    = layers.Reshape((1,1,-1))(layers.Lambda(lambda x: K.min(x, axis=1, keepdims=True))(layers.Reshape((BT_K*BT_K, -1))(next_board)))
    #scale_state  = layers.Lambda(lambda x: K.switch(K.not_equal(x[0], x[1]), x[0] - x[1], K.ones_like(x[0])))([max_state, min_state])
    #next_board   = layers.Lambda(lambda x: (x[0] - x[1])/x[2])([next_board, min_state, scale_state]) # renormalize

    reward = layers.Flatten()(x)
    reward = layers.Dense((SUPPORT_SHAPE), activation='softmax', name='reward', kernel_regularizer=l2(WEIGHT_DECAY), bias_regularizer=l2(WEIGHT_DECAY))(reward)

    return keras.Model([input_board, input_action], outputs={"next_board": next_board, "reward": reward}, name="Dynamics")

@tf.custom_gradient
def scale_grad_layer(x):
    def grad(dy):
        return dy / 2.0
    return tf.identity(x), grad


def unroll_networks(state_network, policy_value_network, dynamics_network):


    input_state  = keras.Input(shape=BOARD_SHAPE, name="starting_board")
    actions      = keras.Input(shape=(N_UNROLL_STEPS,)+ACTION_SHAPE, name="actions")

    hidden_state = state_network(input_state)

    policies, values, rewards = [], [], []

    for i in range(N_UNROLL_STEPS):
        res = policy_value_network(hidden_state)
        policy, value         = res['policy'], res['value']
        res = dynamics_network([hidden_state, actions[:,i]])
        hidden_state, reward  =  scale_grad_layer(res['next_board']), res['reward']

        policies.append(layers.Reshape((1,)+ACTION_SHAPE, name="policy" if N_UNROLL_STEPS == 1 else "p_"+str(i))(policy))
        values.append(layers.Reshape((1,SUPPORT_SHAPE), name="value" if N_UNROLL_STEPS == 1 else "v_"+str(i))(value))
        rewards.append(layers.Reshape((1, SUPPORT_SHAPE), name="reward" if N_UNROLL_STEPS == 1 else "r_"+str(i))(reward))
    
    if N_UNROLL_STEPS > 1:
        policy = layers.Concatenate(axis=1, name="policy")(policies)
        value  = layers.Concatenate(axis=1, name="value")(values)
        reward = layers.Concatenate(axis=1, name="reward")(rewards)
    else:
        policy = policies[0]
        value = values[0]
        reward = rewards[0]
    return keras.Model(inputs={"board": input_state, "actions": actions}, outputs={"policy": policy, "value": value, "reward": reward}, name="Unrolled")
