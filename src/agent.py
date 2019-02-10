import gym
import numpy as np

from replay_buffer import ReplayBuffer

class AntAgent:

    def __init__(self, render=False, model=None):
        # create an environment
        self.environment = gym.make('MountainCarContinuous-v0') 
        # reset environment when an agent is initialized
        self.current_observation = self.reset_environment()
        self.render = render
        self.model = model

        self.buffer = ReplayBuffer()


    def reset_environment(self):
        current_observation = self.environment.reset()
        return current_observation

    def get_action(self, current_observation):
        """Fetch an action according to model policy"""
        if self.model is None:
            action = self.environment.action_space.sample()
        else:
            action = self.model.predict(current_observation)
        
        return action

    def get_transitions(self, action):
        """Take one step in the environment and return the observations"""
        next_observation, reward, done, _ = self.environment.step(action)
        
        if self.render:
            self.environment.render()
        return next_observation, reward, done

    def run_episode(self, num_episodes=1):
        """run episodes `num_episodes` times using `model` policy"""
        for episode in range(num_episodes):
            self.current_observation = self.reset_environment()
            episode_id = self.buffer.create_episode()

            done=False
            transition = dict()
            
            while not done:
                transition['current_observation'] = self.current_observation
                transition['action'] = self.get_action(self.current_observation)
                transition['next_observation'], transition['reward'], done = self.get_transitions(transition['action'])

                self.buffer.add_sample(episode_id, transition)

            self.buffer.add_episode(episode_id)

                # self.store_transition()

if __name__=="__main__":
    agent = AntAgent()
    agent.run_episode(num_episodes=2)