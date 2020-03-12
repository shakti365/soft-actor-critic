import random
import tensorflow as tf
import numpy as np

from agent import Agent
from sac import SAC

num_episodes = 10
num_steps = 100

# Initialise policy
config = dict()
config['epochs'] = 1000
config['learning_rate'] = 0.0001
config['target_update'] = 10
config['gamma'] = 0.9
config['model_name'] = 'experiments'
config['seed'] = 42
config['log_step'] = 1
config['train_batch_size'] = 128
config['valid_batch_size'] = 128
config['optimizer'] = 'adam'
config['initializer'] = 'uniform'
config['logs_path'] = '../data/models'
config['split'] = 0.8

# Initialise agent, environment and policy
# Initialize replay memory
render = False
agent = Agent(render=render, model=None)
model = SAC(config)

# Clear default graph stack and reset global graph definition.
tf.reset_default_graph()

# Set seed for random.
tf.set_random_seed(model.seed)

# Create input placeholders.
current_states = tf.placeholder(shape=[None, 2], dtype=tf.float32)
actions = tf.placeholder(shape=[None, 1], dtype=tf.int32)
rewards = tf.placeholder(shape=[None, 1], dtype=tf.float32)
next_states = tf.placeholder(shape=[None, 2], dtype=tf.float32)

# Get loss and optimization ops
optimize_op, v_loss_op, q_loss_op, policy_loss_op, summary = model.train(current_states, actions, rewards, next_states)

log_pi, action = model.log_policy(current_states)

# Create model copy op.
copy_op = model.copy(primary_scope='primary', target_scope='target')

# Object to saver model checkpoints
saver = tf.train.Saver()

global_step = 0
total_reward = 0
exploration = 0
exploitation = 0

with tf.Session() as sess:

    # Create file writer directory to store summary and events.
    train_writer = tf.summary.FileWriter(model.TF_SUMMARY_DIR+'/train', sess.graph)
    valid_writer = tf.summary.FileWriter(model.TF_SUMMARY_DIR+'/valid')

    # Initialize parameter vectors
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # In every iteration:
    for episode in range(num_episodes):
        
        episode_id = agent.buffer.create_episode()
        done = False
        current_observation = agent.reset_environment()
        
        # In each step:
        while not done:

            transition = dict()
            transition['current_observation'] = current_observation

            current_epsilon = 0.1
            # Epsilon greedy policy
            if (random.uniform(0, 1) < current_epsilon) or (global_step == 0):
                # Get a random action.
                transition['action'] = agent.get_action(current_observation,
                                                       random=True)
                exploration += 1
            else:
                # Get recommended action from the policy
                transition['action'] = sess.run(action, {current_states:
                                                      transition['current_observation'].reshape(-1,2)})
                #logger.log(q_values_)
                exploitation += 1

            # Observe next state and reward
            transition['next_observation'], transition['reward'], done = agent.get_transitions(transition['action'])
            if done is True:
                transition['end'] = 0.0
            else:
                transition['end'] = 1.0

            # Add this transition to the replay buffer
            agent.buffer.add_sample(episode_id, transition)

        agent.buffer.add_episode(episode_id)

        # In every gradient step:
        for step in range(num_steps):

            # Sample array of transitions from replay buffer.
            transition_matrices = agent.buffer.fetch_sample()

            feed_dict ={
                current_states: transition_matrices[0].astype(np.float32),
                actions: transition_matrices[1].astype(np.int32),
                rewards: transition_matrices[2].astype(np.float32),
                next_states: transition_matrices[3].astype(np.float32),
                # end: transition_matrices[4].astype(np.float32)
            }

            train_loss, train_loss_2, train_loss_3, train_summary, _ = sess.run([v_loss_op, q_loss_op, policy_loss_op, summary, optimize_op], feed_dict)
            print (episode, step, train_loss, train_loss_2, train_loss_3)

            # Log training dataset.
            train_writer.add_summary(train_summary, step)

            # Check if step to update Q target.
            if step % model.target_update == 0:
                sess.run(copy_op)

            global_step += 1
