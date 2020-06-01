import numpy as np
from sac import Critic
import tensorflow as tf

tf.random.set_seed(0)
critic = Critic()

state = np.ones([2, 3])
action = np.ones([2, 1])
q = critic.forward(state, action)
print(state, action, q)
