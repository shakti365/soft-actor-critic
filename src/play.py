import gym
import argparse
import logging
from datetime import datetime
import numpy as np

from sac import Actor

logging.basicConfig(level='INFO')

parser = argparse.ArgumentParser(description='SAC')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')
parser.add_argument('--env_name', type=str, default='MountainCarContinuous-v0',
                    help='name of the gym environment with version')
parser.add_argument('--render', type=bool, default=False,
                    help='set gym environment to render display')
parser.add_argument('--verbose', type=bool, default=False,
                    help='log execution details')
parser.add_argument('--model_path', type=str, default='../data/models/',
                    help='path to save model')
parser.add_argument('--model_name', type=str,
                    default=f'{str(datetime.utcnow().date())}-{str(datetime.utcnow().time())}',
                    help='name of the saved model')

while True:

    args = parser.parse_args()


    # Instantiate the environment.
    env = gym.make(args.env_name)
    env.seed(args.seed)
    state_space = env.observation_space.shape[0]
    # TODO: fix this when env.action_space is not `Box`
    action_space = env.action_space.shape[0]

    actor = Actor(action_space)

    actor.load_weights(args.model_path + args.model_name + '/model')

    # Observe state
    current_state = env.reset()

    episode_reward = 0
    done = False
    while not done:

        if args.render:
            env.render()

        current_state_ = np.array(current_state, ndmin=2)
        action_, _ = actor(current_state_)
        action = action_.numpy()[0]

        # Execute action, observe next state and reward
        next_state, reward, done, _ = env.step(action)

        episode_reward +=  reward

        # Update current state
        current_state = next_state

    print(episode_reward)
