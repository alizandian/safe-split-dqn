from MapInfo import MapInfo
from Animation import Animation
from GameEngine import GameEngine
import tensorflow as tf
from tf_agents.specs import tensor_spec
from tf_agents.environments import tf_py_environment
import sys
import numpy as np


def test_basic_operation():
    print("------------------ Test 1: Env --------------------")
    test = GameEngine()
    time_step = test.reset()
    print(time_step)
    cumulative_reward = time_step.reward

    # dummy action
    action1 = np.zeros(15)

    # 3 iterations to test the data transition
    for _ in range(3):
        time_step = test.step(action1)
        print(time_step)
        cumulative_reward += time_step.reward


def test_action_spec():
    print("------------------ Test 2: Spec --------------------")
    mapinfo = MapInfo()
    env = GameEngine()
    print("action spec")
    action_tensor_spec = tensor_spec.from_spec(env.action_spec())
    print(env.action_spec())
    print(action_tensor_spec)
    num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1
    print(num_actions)

    # convert the pyEnv into TFPyEnv
    tf_env = tf_py_environment.TFPyEnvironment(env)
    time_step = tf_env.reset()
    print(time_step)
    
    # the blow does not work...
    # action = tf.random.uniform([15,], 0, 2, dtype=tf.int32)  
    # time_step = tf_env.step(action)
    # print(time_step)


def test_animation():
    print("------------------ Test 3: Animation --------------------")
    mapinfo = MapInfo()
    canvas = Animation(mapinfo.get_geometric_info())
    state1 = np.array([0,1,2,3,4,5,0,1,2,3,4,5,0,0], dtype=np.int32)
    state2 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype=np.int32)
    state3 = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1], dtype=np.int32)
    state4 = np.array([2,2,2,2,2,2,2,2,2,2,2,2,2,2], dtype=np.int32)
    state5 = np.array([3,3,3,3,3,3,3,3,3,3,3,3,3,3], dtype=np.int32)
    states = [state1, state2, state3, state4, state5]
    
    action1 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype=np.int32)
    action2 = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], dtype=np.int32)
    action3 = np.array([2,2,2,2,2,2,2,2,2,2,2,2,2,2,2], dtype=np.int32)
    action4 = np.array([0,1,2,0,1,2,0,1,2,0,1,2,0,1,2], dtype=np.int32)
    action5 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype=np.int32)
    actions = [action1, action2, action3, action4, action5] 

    canvas.playback_animation(states, actions)
    

if __name__ == "__main__":
    test_basic_operation()
    test_action_spec()
    test_animation()
