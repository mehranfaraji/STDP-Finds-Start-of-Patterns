import os
import yaml

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse

from InputSimulator import make_input, make_pattern_presentation_array, copy_and_paste_jittered_pattern
from InputSimulator import triple_input_runtime, add_noise


with open("hyperparameters.yml", "r") as yaml_file:
    hyperparameters = yaml.safe_load(yaml_file)

input_folder = hyperparameters['save_input']
tripling = hyperparameters["tripling"]
is_continuous = hyperparameters["is_continuous"]
runduration = hyperparameters["runduration"]
dt = hyperparameters["dt"]
num_neurons = hyperparameters["num_neurons"]
number_pat = hyperparameters["number_pat"]
max_time_wo_spike = hyperparameters["max_time_wo_spike"]
min_rate = hyperparameters["min_rate"]
max_rate = hyperparameters["max_rate"]
max_change_speed = hyperparameters["max_change_speed"]
patternlength = hyperparameters["patternlength"]
pattern_freq = hyperparameters["pattern_freq"]
max_rate_add = hyperparameters["max_rate_add"]
min_rate_add = hyperparameters["min_rate_add"]
max_time_wo_spike_add = hyperparameters["max_time_wo_spike_add"]
max_change_speed_add = hyperparameters["max_change_speed_add"]
spike_del = hyperparameters["spike_del"]
jitter_sd = hyperparameters["jitter_sd"]


def get_spiketrain(times, indices, num_neurons, runduration, dt):
    spike_train = np.zeros((num_neurons, int(runduration/dt/3)))
    for time, index in zip(times, indices):
        time = int(time/dt)
        if time == spike_train.shape[1]:
            break
        spike_train[index,time] = 1
    return spike_train

def get_pattern_times(position_copypaste, patternlength, runduration):
    r = np.arange(0,runduration, patternlength)
    pattern_times = position_copypaste * r
    pattern_times = [0.0] + [t for t in pattern_times if t !=0]
    return pattern_times

def generate_input_data():
    
    indices, times = make_input(min_rate, max_rate, max_time_wo_spike, max_change_speed, runduration, num_neurons, dt, is_continuous)
    position_copypaste = make_pattern_presentation_array(runduration, patternlength, pattern_freq)

    indices, times, (indices_pattern, times_pattern) = copy_and_paste_jittered_pattern(times, indices, position_copypaste, patternlength, jitter_sd, spike_del,number_pat)
    indices_add, times_add = make_input(min_rate_add, max_rate_add, max_time_wo_spike_add,
                                    max_change_speed_add, runduration, num_neurons, dt, is_continuous)
    times, indices = add_noise(times, indices, times_add, indices_add)
    indices = indices.astype(int)
    if tripling and runduration > 300:
            spike_train = get_spiketrain(times, indices, num_neurons, runduration, dt)
            spike_train = np.concatenate((spike_train, spike_train, spike_train), axis=1)
            times, indices = triple_input_runtime(times, indices)
            position_copypaste = np.concatenate((position_copypaste, position_copypaste, position_copypaste))
    else:
        spike_train = get_spiketrain(times, indices, num_neurons, runduration, dt)
    
    pattern_times = get_pattern_times(position_copypaste, patternlength, runduration)

    if not os.path.exists(input_folder):
        os.makedirs(input_folder)

    np.save(os.path.join(input_folder, "times.npy"), times)
    np.save(os.path.join(input_folder, "indices.npy"), indices)
    np.save(os.path.join(input_folder, "times_pattern.npy"), times_pattern)
    np.save(os.path.join(input_folder, "indices_pattern.npy"), indices_pattern)
    np.save(os.path.join(input_folder, "position_copypaste.npy"), position_copypaste)
    sparse_spike_train = sparse.csr_matrix(spike_train)
    sparse.save_npz(os.path.join(input_folder, "sparse_spike_train.npz"), sparse_spike_train)
    np.save(os.path.join(input_folder, "pattern_times.npy"), pattern_times)

if __name__ == "__main__":
    generate_input_data()