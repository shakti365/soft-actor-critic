import numpy as np
from sac import Actor
import tensorflow as tf

tf.random.set_seed(0)
actor = Actor(1)

inp = np.ones([2, 3])
out = actor.forward(inp)
print(inp, out)
