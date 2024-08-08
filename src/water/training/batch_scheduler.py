import numpy as np


class BatchSizeScheduler:
    def __init__(self, patience=20, step_size=1, mode='max', initial_iterations=1, max_iterations=10000, **kwargs):
        self.mode = mode
        self.patience = patience
        self.step_size = step_size
        self.initial_iterations = initial_iterations
        self.required_iterations = initial_iterations
        self.current_iteration = 0
        self.current_best = -np.inf if mode == 'max' else np.inf
        self.num_bad_epochs = 0
        self.max_iterations = max_iterations

    def state_dict(self):
        return self.__dict__

    def load_state_dict(self, state_dict: dict):
        self.__dict__.update(state_dict)

    def is_min_max(self, value):
        match self.mode:
            case 'min':
                return min(value, self.current_best) == value
            case 'max':
                return max(value, self.current_best) == value

    def check_count(self):
        if self.num_bad_epochs > self.patience:
            self.num_bad_epochs = 0
            self.required_iterations += self.step_size
            self.required_iterations = int(min(self.required_iterations + self.step_size, self.max_iterations))

    def next_iteration(self):
        self.current_iteration = (1 + self.current_iteration) % self.required_iterations

    def step(self, value):
        if self.is_min_max(value):
            self.current_best = value
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
            self.check_count()
        self.next_iteration()
