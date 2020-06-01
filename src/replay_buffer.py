import numpy as np

class ReplayBuffer:

    def __init__(self, state_space, action_space, max_size=100000):
        self.current_states = np.empty((0, state_space), dtype=np.float64)
        self.actions = np.empty((0, action_space), dtype=np.float64)
        self.rewards = np.empty((0, 1), dtype=np.float64)
        self.next_states = np.empty((0, state_space), dtype=np.float64)
        self.ends = np.empty((0, 1), dtype=np.float64)
        self.total_size = 0
        self.max_size = max_size

    def store(self, current_state, action, reward, next_state, end):
        self.current_states = np.append(self.current_states[-self.max_size:],
                                        np.array(current_state, ndmin=2), axis=0)
        self.actions = np.append(self.actions[-self.max_size:],
                                 np.array(action, ndmin=2), axis=0)
        self.rewards = np.append(self.rewards[-self.max_size:],
                                 np.array(reward, ndmin=2), axis=0)
        self.next_states = np.append(self.next_states[-self.max_size:],
                                     np.array(next_state, ndmin=2), axis=0)
        self.ends = np.append(self.ends[-self.max_size:],
                              np.array(end, ndmin=2), axis=0)
        self.total_size += 1

    def fetch_sample(self, num_samples):

        if num_samples > self.total_size:
            num_samples = self.total_size

        idx = np.random.choice(range(min(self.total_size, self.max_size)),
                               size=num_samples,
                               replace=False)

        current_states_ = self.current_states[idx]
        actions_ = self.actions[idx]
        rewards_ = self.rewards[idx]
        next_states_ = self.next_states[idx]
        ends_ = self.ends[idx]

        return current_states_, actions_, rewards_, next_states_, ends_
