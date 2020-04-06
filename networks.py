
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.regularizers import l2 

from settings import *

def residual_block(input, name, size=32, activation='relu', convert=False, weight_decay=1e-5):
    if convert:
        input = layers.Conv2D(size, 3, padding='same', name="res_conv_trans_{}".format(name),
                        kernel_regularizer=l2(weight_decay))(input)
        input = layers.Activation(activation)(input)
        input = layers.BatchNormalization()(input)
    
    X_skip = input
    x1 = layers.Conv2D(size, 3, padding='same', name="res_conv_A_{}".format(name),
                        kernel_regularizer=l2(weight_decay))(input)
    x1 = layers.Activation(activation)(x1)
    x1 = layers.BatchNormalization()(x1)

    x2 = layers.Conv2D(size, 3, padding='same', name="res_conv_B_{}".format(name),
                        kernel_regularizer=l2(weight_decay))(x1)
    x2 = layers.Activation(activation)(x2)
    x2 = layers.BatchNormalization()(x2)

    return layers.Add()([x2, X_skip])

def policy_value_network_alpha(config):
    board_shape = get_board_shape(config)
    action_shape = get_action_shape(config)

    BT_K = board_shape[1]

    input   = keras.Input(shape=board_shape, name='board')
    x       = layers.Reshape(board_shape)(input) # assert board shape
    x       = layers.Permute((2,3,1,4))(x)
    x       = layers.Reshape((BT_K, BT_K, -1))(x)

    x       = residual_block(x, "pv_a", convert=True)
    x       = residual_block(x, "pv_b")
    x       = residual_block(x, "pv_c")
    
    policy  = residual_block(x, "pv_d", size=action_shape[-1], convert=True)
    policy  = layers.Flatten()(policy)
    policy  = layers.Activation(activation='softmax')(policy)
    policy  = layers.Reshape(action_shape, name='policy')(policy)

    value   = residual_block(x, "pv_e")
    value   = layers.Flatten()(value)
    value   = layers.Dense((1), activation='sigmoid', name='value', kernel_regularizer=l2(config.training.weight_decay), bias_regularizer=l2(config.training.weight_decay))(value)

    return keras.Model(inputs=input, outputs={"policy": policy, "value": value})

def prediction_network_mu(config):
    hidden_shape = config.mu.repr_shape
    action_shape = get_action_shape(config)

    input   = keras.Input(shape=hidden_shape, name='board')
    x       = layers.Reshape(hidden_shape, name='PredictionNetworkBoard')(input)

    x       = residual_block(x, "pred_a", convert=True)
    x       = residual_block(x, "pred_b")
    x       = residual_block(x, "pred_c")
    policy  = residual_block(x, "pred_d", size=action_shape[-1], convert=True)

    policy  = layers.Flatten()(policy)
    if len(action_shape) == 1:
        policy = layers.Dense(action_shape[0], kernel_regularizer=l2(config.training.weight_decay), bias_regularizer=l2(config.training.weight_decay))(policy)
    policy  = layers.Activation(activation='softmax')(policy)
    policy  = layers.Reshape(action_shape, name='policy')(policy)

    value   = residual_block(x, "pred_e")
    value   = layers.Flatten()(value)
    value   = layers.Dense((2*config.mu.puct.value_support+1), activation='softmax', name='value', kernel_regularizer=l2(config.training.weight_decay), bias_regularizer=l2(config.training.weight_decay))(value)

    return keras.Model(inputs=input, outputs={"policy": policy, "value": value}, name="Prediction")

def representation_network(config):
    if config.game.kind == "Gym":
        return representation_network_atari(config)
    else:
        board_shape = get_board_shape(config)
        hidden_shape = config.mu.repr_shape

        BT_K = board_shape[1]

        input   = keras.Input(shape=board_shape, name='board')
        x       = layers.Reshape(board_shape)(input) # assert board shape
        x       = layers.Permute((2,3,1,4))(x)
        x       = layers.Reshape((BT_K, BT_K, board_shape[0]*board_shape[3]), name='RepresentationNetworkBoard')(x)

        x       = residual_block(x, "repr_a", convert=True)
        x       = residual_block(x, "repr_b")
        x       = residual_block(x, "repr_c")

        repr_board   = layers.Conv2D(hidden_shape[-1], (3, 3), padding='same', activation='relu', kernel_regularizer=l2(config.training.weight_decay), bias_regularizer=l2(config.training.weight_decay), name='repr_board')(x)

        return keras.Model(inputs=input, outputs=repr_board, name="Representation")

def representation_network_atari(config):
    board_shape = get_board_shape(config)
    hidden_shape = config.mu.repr_shape

    BT_K = board_shape[1]

    input   = keras.Input(shape=board_shape, name='board')
    x       = layers.Reshape(board_shape)(input) # assert board shape
    x       = layers.Permute((2,3,1,4))(x)
    x       = layers.Reshape((BT_K,BT_K,board_shape[0]*board_shape[3]))(x)
    x       = layers.Conv2D(32, 3, padding='same', strides=2)(x)
    x       = residual_block(x, "repr_a", size=32)
    x       = layers.Conv2D(64, 3, padding='same', strides=2)(x)
    x       = residual_block(x, "repr_b", size=64)
    x       = layers.AveragePooling2D()(x)
    x       = residual_block(x, "repr_c", size=64)
    x       = layers.AveragePooling2D()(x)
    x       = residual_block(x, "repr_d", size=64)
    repr_board   = layers.Conv2D(hidden_shape[-1], (3, 3), padding='same', activation='relu', kernel_regularizer=l2(config.training.weight_decay), bias_regularizer=l2(config.training.weight_decay), name='repr_board')(x)

    return keras.Model(inputs=input, outputs=repr_board, name="Representation")


def dynamics_network(config):
    action_shape = get_action_shape(config)
    hidden_shape = config.mu.repr_shape

    input_board  = keras.Input(shape=hidden_shape, name='board')
    input_board_  = layers.Reshape(hidden_shape, name='DynamicsNetworkState')(input_board)
    input_action = keras.Input(shape=action_shape, name='action')
    input_action_  = layers.Reshape(action_shape, name='DynamicsNetworkAction')(input_action)
    if config.game.kind == "Gym":
        input_action_ = layers.RepeatVector(hidden_shape[0]**2)(input_action_)
        input_action_ = layers.Reshape((hidden_shape[0], hidden_shape[1])+action_shape)(input_action_)
    
    x            = layers.Concatenate(axis=-1)([input_board_, input_action_])#, (BT_K, BT_K, HIDDEN_PLANES + ACTION_PLANES))

    x       = residual_block(x, "dyn_a", convert=True)
    x       = residual_block(x, "dyn_b")
    x       = residual_block(x, "dyn_c")

    next_board    = residual_block(x, "dyn_d", size=hidden_shape[-1], convert=True)

    reward = layers.Flatten()(x)
    reward = layers.Dense((2*config.mu.reward_support+1), activation='softmax', name='reward', kernel_regularizer=l2(config.training.weight_decay), bias_regularizer=l2(config.training.weight_decay))(reward)

    return keras.Model([input_board, input_action], outputs={"next_board": next_board, "reward": reward}, name="Dynamics")

@tf.custom_gradient
def scale_grad_layer(x):
    def grad(dy):
        return dy / 2.0
    return tf.identity(x), grad


def unroll_networks(config, state_network, policy_value_network, dynamics_network):
    board_shape = get_board_shape(config)
    action_shape = get_action_shape(config)
    hidden_shape = config.mu.repr_shape
    unroll_steps = config.mu.unroll_steps

    input_state  = keras.Input(shape=board_shape, name="starting_board")
    actions      = keras.Input(shape=(unroll_steps,)+action_shape, name="actions")

    hidden_state = state_network(input_state)

    policies, values, rewards = [], [], []

    for i in range(unroll_steps):
        res = policy_value_network(hidden_state)
        policy, value         = res['policy'], res['value']
        res = dynamics_network([hidden_state, actions[:,i]])
        hidden_state, reward  =  scale_grad_layer(res['next_board']), res['reward']

        policies.append(layers.Reshape((1,)+action_shape, name="policy" if unroll_steps == 1 else "p_"+str(i))(policy))
        values.append(layers.Reshape((1,2*config.mu.puct.value_support+1), name="value" if unroll_steps == 1 else "v_"+str(i))(value))
        rewards.append(layers.Reshape((1, 2*config.mu.reward_support+1), name="reward" if unroll_steps == 1 else "r_"+str(i))(reward))
    
    if unroll_steps > 1:
        policy = layers.Concatenate(axis=1, name="policy")(policies)
        value  = layers.Concatenate(axis=1, name="value")(values)
        reward = layers.Concatenate(axis=1, name="reward")(rewards)
    else:
        policy = policies[0]
        value = values[0]
        reward = rewards[0]
    return keras.Model(inputs={"board": input_state, "actions": actions}, outputs={"policy": policy, "value": value, "reward": reward}, name="Unrolled")
