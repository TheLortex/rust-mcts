
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.regularizers import l2 

from settings import *


def policy_value_network_alpha():
    input   = keras.Input(shape=BOARD_SHAPE, name='board')
    x       = layers.Reshape((BT_K, BT_K, HISTORY_LENGTH*3))(input)
    x       = layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(WEIGHT_DECAY), bias_regularizer=l2(WEIGHT_DECAY))(x)
    x       = layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(WEIGHT_DECAY), bias_regularizer=l2(WEIGHT_DECAY))(x)
    x       = layers.Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(WEIGHT_DECAY), bias_regularizer=l2(WEIGHT_DECAY))(x)
    
    policy  = layers.Conv2D(3, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(WEIGHT_DECAY), bias_regularizer=l2(WEIGHT_DECAY))(x)
    policy  = layers.Flatten()(policy)
    policy  = layers.Activation(activation='softmax')(policy)
    policy  = layers.Reshape(ACTION_SHAPE, name='policy')(policy)

    value   = layers.Flatten()(x)
    value   = layers.Dense((2*SUPPORT_SIZE+1), activation='softmax', name='value', kernel_regularizer=l2(WEIGHT_DECAY), bias_regularizer=l2(WEIGHT_DECAY))(value)

    return keras.Model(inputs=input, outputs={"policy": policy, "value": value})

def prediction_network_mu():
    input   = keras.Input(shape=HIDDEN_SHAPE, name='board')
    x       = layers.Reshape(HIDDEN_SHAPE)(input)
    x       = layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(WEIGHT_DECAY), bias_regularizer=l2(WEIGHT_DECAY))(x)
    x       = layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(WEIGHT_DECAY), bias_regularizer=l2(WEIGHT_DECAY))(x)
    x       = layers.Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(WEIGHT_DECAY), bias_regularizer=l2(WEIGHT_DECAY))(x)
    #x       = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    #x       = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    #x       = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    #policy  = layers.Dense(ACTION_SHAPE, activation='softmax', name='policy')(x)
    policy  = layers.Conv2D(3, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(WEIGHT_DECAY), bias_regularizer=l2(WEIGHT_DECAY))(x)
    policy  = layers.Flatten()(policy)
    policy  = layers.Activation(activation='softmax')(policy)
    policy  = layers.Reshape(ACTION_SHAPE, name='policy')(policy)

    value   = layers.Flatten()(x)
    value   = layers.Dense((2*SUPPORT_SIZE+1), activation='sigmoid', name='value', kernel_regularizer=l2(WEIGHT_DECAY), bias_regularizer=l2(WEIGHT_DECAY))(value)

    return keras.Model(inputs=input, outputs={"policy": policy, "value": value})

def representation_network():
    input   = keras.Input(shape=BOARD_SHAPE, name='board')
    x       = layers.Reshape((BT_K, BT_K, HISTORY_LENGTH*3))(input)
    x       = layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(WEIGHT_DECAY), bias_regularizer=l2(WEIGHT_DECAY))(x)
    x       = layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(WEIGHT_DECAY), bias_regularizer=l2(WEIGHT_DECAY))(x)
    x       = layers.Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(WEIGHT_DECAY), bias_regularizer=l2(WEIGHT_DECAY))(x)
    repr_board   = layers.Conv2D(HIDDEN_PLANES, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(WEIGHT_DECAY), bias_regularizer=l2(WEIGHT_DECAY), name='repr_board')(x)

    return keras.Model(inputs=input, outputs=repr_board)

def dynamics_network():
    input_board  = keras.Input(shape=HIDDEN_SHAPE, name='board')
    input_board_  = layers.Reshape(HIDDEN_SHAPE)(input_board)
    input_action = keras.Input(shape=ACTION_SHAPE, name='action')
    input_action_  = layers.Reshape(ACTION_SHAPE)(input_action)
    x            = layers.Concatenate(axis=-1)([input_board_, input_action_])#, (BT_K, BT_K, HIDDEN_PLANES + ACTION_PLANES))
    x            = layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(WEIGHT_DECAY), bias_regularizer=l2(WEIGHT_DECAY))(x)
    x            = layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(WEIGHT_DECAY), bias_regularizer=l2(WEIGHT_DECAY))(x)
    x            = layers.Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(WEIGHT_DECAY), bias_regularizer=l2(WEIGHT_DECAY))(x)

    next_board    = layers.Conv2D(HIDDEN_PLANES, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(WEIGHT_DECAY), bias_regularizer=l2(WEIGHT_DECAY))(x)
    max_state    = layers.Reshape((1,1,-1))(layers.Lambda(lambda x: K.max(x, axis=1, keepdims=True))(layers.Reshape((BT_K*BT_K, -1))(next_board)))
    min_state    = layers.Reshape((1,1,-1))(layers.Lambda(lambda x: K.min(x, axis=1, keepdims=True))(layers.Reshape((BT_K*BT_K, -1))(next_board)))
    scale_state  = layers.Lambda(lambda x: K.switch(K.not_equal(x[0], x[1]), x[0] - x[1], K.ones_like(x[0])))([max_state, min_state])
    next_board   = layers.Lambda(lambda x: (x[0] - x[1])/x[2])([next_board, min_state, scale_state]) # renormalize

    reward = layers.Flatten()(x)
    reward = layers.Dense((SUPPORT_SHAPE), activation='sigmoid', name='reward', kernel_regularizer=l2(WEIGHT_DECAY), bias_regularizer=l2(WEIGHT_DECAY))(reward)

    return keras.Model(inputs={"board": input_board, "action": input_action}, outputs={"next_board": next_board, "reward": reward})

@tf.custom_gradient
def scale_grad_layer(x):
    def grad(dy):
        return dy / 2.0
    return tf.identity(x), grad


def unroll_networks(state_network, policy_value_network, dynamics_network):


    input_state  = keras.Input(shape=BOARD_SHAPE)
    actions      = keras.Input(shape=(N_UNROLL_STEPS,)+ACTION_SHAPE)

    hidden_state = state_network(input_state)

    policies, values, rewards = [], [], []

    for i in range(N_UNROLL_STEPS):
        res = policy_value_network(hidden_state)
        policy, value         = res['policy'], res['value']
        res = dynamics_network({'board': hidden_state, 'action': actions[:,i]})
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
    return keras.Model(inputs={"board": input_state, "actions": actions}, outputs={"policy": policy, "value": value, "reward": reward})
