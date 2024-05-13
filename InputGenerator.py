import os
import yaml

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse

from InputSimulator import make_input, make_pattern_presentation_array, copy_and_paste_jittered_pattern
from InputSimulator import triple_input_runtime, add_noise


class InputGenerator():
    def __init__(self, hyperparameter_path="hyperparameters.yml"):
        with open(hyperparameter_path, "r") as yaml_file:
            hyperparameters = yaml.safe_load(yaml_file)

        self.input_folder = hyperparameters['save_input']
        self.tripling = hyperparameters["tripling"]
        self.is_continuous = hyperparameters["is_continuous"]
        self.runduration = hyperparameters["runduration"]
        self.dt = hyperparameters["dt"]
        self.num_neurons = hyperparameters["num_neurons"]
        self.number_pat = hyperparameters["number_pat"]
        self.max_time_wo_spike = hyperparameters["max_time_wo_spike"]
        self.min_rate = hyperparameters["min_rate"]
        self.max_rate = hyperparameters["max_rate"]
        self.max_change_speed = hyperparameters["max_change_speed"]
        self.patternlength = hyperparameters["patternlength"]
        self.pattern_freq = hyperparameters["pattern_freq"]
        self.max_rate_add = hyperparameters["max_rate_add"]
        self.min_rate_add = hyperparameters["min_rate_add"]
        self.max_time_wo_spike_add = hyperparameters["max_time_wo_spike_add"]
        self.max_change_speed_add = hyperparameters["max_change_speed_add"]
        self.spike_del = hyperparameters["spike_del"]
        self.jitter_sd = hyperparameters["jitter_sd"]


    def get_spiketrain(self, times, indices, num_neurons, runduration, dt):
        spike_train = np.zeros((num_neurons, int(runduration/dt/3)))
        for time, index in zip(times, indices):
            time = int(time/dt)
            if time == spike_train.shape[1]:
                break
            spike_train[index,time] = 1
        return spike_train

    def get_pattern_times(self, position_copypaste, patternlength, runduration):
        r = np.arange(0,runduration, patternlength)
        pattern_times = position_copypaste * r
        pattern_times = [0.0] + [t for t in pattern_times if t !=0]
        return pattern_times

    def update_attributes(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")
    
    def generate_input_data(self, **kwargs):
        self.update_attributes(**kwargs)
        
        indices, times = make_input(self.min_rate, self.max_rate, self.max_time_wo_spike, self.max_change_speed, self.runduration, self.num_neurons, self.dt, self.is_continuous)
        position_copypaste = make_pattern_presentation_array(self.runduration, self.patternlength, self.pattern_freq)

        indices, times, (indices_pattern, times_pattern) = copy_and_paste_jittered_pattern(times, indices, position_copypaste, self.patternlength, self.jitter_sd, self.spike_del, self.number_pat)
        indices_add, times_add = make_input(self.min_rate_add, self.max_rate_add, self.max_time_wo_spike_add,
                                        self.max_change_speed_add, self.runduration, self.num_neurons, self.dt, self.is_continuous)
        times, indices = add_noise(times, indices, times_add, indices_add)
        indices = indices.astype(int)
        if self.tripling and self.runduration > 300:
                spike_train = self.get_spiketrain(times, indices, self.num_neurons, self.runduration, self.dt)
                spike_train = np.concatenate((spike_train, spike_train, spike_train), axis=1)
                times, indices = triple_input_runtime(times, indices)
                position_copypaste = np.concatenate((position_copypaste, position_copypaste, position_copypaste))
        else:
            spike_train = self.get_spiketrain(times, indices, self.num_neurons, self.runduration, self.dt)
        
        pattern_times = self.get_pattern_times(position_copypaste, self.patternlength, self.runduration)

        if not os.path.exists(self.input_folder):
            os.makedirs(self.input_folder)

        np.save(os.path.join(self.input_folder, "times.npy"), times)
        np.save(os.path.join(self.input_folder, "indices.npy"), indices)
        np.save(os.path.join(self.input_folder, "times_pattern.npy"), times_pattern)
        np.save(os.path.join(self.input_folder, "indices_pattern.npy"), indices_pattern)
        np.save(os.path.join(self.input_folder, "position_copypaste.npy"), position_copypaste)
        sparse_spike_train = sparse.csr_matrix(spike_train)
        sparse.save_npz(os.path.join(self.input_folder, "sparse_spike_train.npz"), sparse_spike_train)
        np.save(os.path.join(self.input_folder, "pattern_times.npy"), pattern_times)

if __name__ == "__main__":
    input_generator = InputGenerator("hyperparameters.yml")
    input_generator.generate_input_data()