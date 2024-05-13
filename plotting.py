import numpy as np
import matplotlib.pyplot as plt

def plot_input(times, indices, params, times_pattern=None, indices_pattern=None, save_path=None):
    plt.figure(figsize=(12,5))
    start_time = params['start_time']
    end_time = params['end_time']
    start_index = params['start_index']
    end_index = params['end_index']

    sampletimes = times[(times < end_time) & (indices < end_index) & (times > start_time) & (indices > start_index)]
    sampleindices = indices[(times < end_time) & (indices < end_index) & (times > start_time) & (indices > start_index)]
    intervals = np.arange(start_time, end_time+0.01, 0.05)
    colors = ['#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c']
    num_colors_needed = len(intervals) - 1
    num_colors_available = len(colors)
    colors = [colors[i % num_colors_available] for i in range(num_colors_needed)]
    for i in range(len(intervals) - 1):
        plt.axvline(intervals[i+1])
        plt.axvspan(intervals[i], intervals[i + 1], facecolor=colors[i], alpha=0.7)
    plt.plot(sampletimes, sampleindices, '.k', alpha=0.6)
    if times_pattern is not None and indices_pattern is not None:
        sampletimespattern = times_pattern[(times_pattern < end_time) & (indices_pattern < end_index) & (times_pattern > start_time) & (indices_pattern > start_index)]
        sampleindicespattern = indices_pattern[(times_pattern < end_time) & (indices_pattern < end_index) & (times_pattern > start_time) & (indices_pattern > start_index)]
        plt.plot(sampletimespattern, sampleindicespattern, '.r', alpha=0.6)
    plt.xlabel('Time (s)')
    plt.ylabel('Neuron number')
    if save_path is not None:
        plt.savefig(save_path)
    # plt.show()


def plot_potential(start_time, end_time, model, position_copypaste, patternlength, dt, save_path=None):
    plt.figure(figsize=(12,5))
    plt.plot(np.arange(start_time,end_time,dt), model.potential_rec[int(start_time/dt):int(end_time/dt)])
    plt.axhline(0, linestyle=':', color='black')
    plt.axhline(model.threshold, linestyle='--', color='red')
    pattern_time = np.where((position_copypaste == 1))[0] * patternlength
    ind = np.where((pattern_time >= start_time) & (pattern_time <= end_time))
    pattern_time[ind]
    for i in pattern_time[ind]:
        plt.axvspan(i, i + patternlength, facecolor='gray', alpha=0.7)
    plt.xlabel('Time (s)')
    plt.ylabel('Membrane Potential (arbitrary units)')
    if save_path is not None:
        plt.savefig(save_path)
    # plt.show()