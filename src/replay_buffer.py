import numpy as np
from collections import defaultdict

class ReplayBuffer:

    def __init__(self):
        self.current_episode_id = 0
        self.step_id = 0

        self.episodes = np.empty((0, 1), dtype=np.float32)
        self.steps = np.empty((0, 1), dtype=np.float32)
        self.observations = np.empty((0, 2), dtype=np.float32)
        self.actions = np.empty((0, 1), dtype=np.float32)
        self.rewards = np.empty((0, 1), dtype=np.float32)
        self.next_observations = np.empty((0, 2), dtype=np.float32)
        self.replay_buffer = dict()


    def add_episode(self, episode_id):
        # Push from local buffer with an id to replay buffer
        mdp = self.replay_buffer[episode_id]
        for step in range(1, len(mdp)):

            observation, action, reward, next_observation = mdp[step]['current_observation'], mdp[step]['action'], mdp[step]['reward'], mdp[step]['next_observation']

            self.episodes = np.append(self.episodes, np.array(episode_id, ndmin=2), axis=0)
            self.steps = np.append(self.steps, np.array(step, ndmin=2), axis=0)
            self.observations = np.append(self.observations, observation.reshape(-1, 2), axis=0)
            self.actions = np.append(self.actions, action.reshape(-1, 1), axis=0)
            self.rewards = np.append(self.rewards, np.array(reward, ndmin=2), axis=0)
            self.next_observations = np.append(self.next_observations, next_observation.reshape(-1, 2), axis=0)

    def add_sample(self, episode_id, transition):
        # create a local buffer for each episode with metadata information
        step = len(self.replay_buffer[episode_id])
        self.replay_buffer[episode_id][step] = transition

    def terminate_episode(self):
        # Increment ID counters
        self.current_episode_id += 1
        self.step_id += 1

        # Reset local buffers
        self.episodes = []
        self.steps = []
        self.observations = []
        self.actions = []
        self.rewards = []
        self.next_observations = []

    def fetch_sample(self, num_samples=1):
        pass

    def save_buffer(self):
        pass

    def create_episode(self):
        self.current_episode_id += 1
        self.replay_buffer[self.current_episode_id] = defaultdict(list)
        return self.current_episode_id

    # def init_local_memory(self):
    #   memory = {
    #       'steps': [],
    #       'observations': [],
    #       'actions': [],
    #       'rewards': [],
    #       'next_observations': []
    #   }
    #   return memory