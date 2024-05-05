import numpy as np
import matplotlib.pyplot as plt
from numba import types, typed
from numba.experimental import jitclass

spec = [
    ('threshold', types.float64),
    ('tau_m', types.float64),
    ('tau_s', types.float64),
    ('K1', types.float64),
    ('K2', types.float64),
    ('dt', types.float64),
    ('tref', types.float64),
    ('A_pos', types.float64),
    ('B', types.float64),
    ('tau_pos', types.float64),
    ('tau_neg', types.float64),
    ('A_neg', types.float64),
    ('last_spike_time', types.float64),
    ('spike_rec', types.ListType(types.float64)),
    ('potential_rec', types.ListType(types.float64)),
    ('ref_counter', types.int64),
    ('afferents_not_spiked_yet', types.optional(types.Array(types.float64, 2, 'C'))),
    ('w_sample', types.optional(types.Array(types.float64, 2, 'C'))),
]

@jitclass(spec)
class SRM():
    def __init__(self, threshold, tau_m, tau_s, K1, K2, dt, tref, A_pos, B, tau_pos, tau_neg) -> None:
        self.threshold = threshold
        self.tau_m = tau_m
        self.tau_s = tau_s
        self.K1 = K1
        self.K2 = K2
        self.dt = dt
        self.tref = tref

        self.A_pos= A_pos
        self.B=B
        self.tau_pos=tau_pos
        self.tau_neg=tau_neg
        self.A_neg = - B * self.A_pos
        
        # initial state of the neuron
        self.last_spike_time = - 10e6
        self.spike_rec = typed.List.empty_list(np.float64)
        self.potential_rec = typed.List([0.0])
        # self.spike_rec = []
        # self.potential_rec = [0.0]
        self.ref_counter = 0
        self.afferents_not_spiked_yet = None
        self.w_sample = None
    
    def do_ltp(self,stdp_window):
        spike_row_index = np.nonzero(stdp_window)
        n , window_length = stdp_window.shape
        afferent_last_spike_time = np.full((n, 1), -np.inf)
        afferent_last_spike_time[spike_row_index[0], 0] = spike_row_index[1]
        afferent_last_spike_time = afferent_last_spike_time * self.dt
        delta_t = afferent_last_spike_time - (window_length) * self.dt
        dw = self.ltp(delta_t)
        return dw 

    def do_ltd(self, n, first_time_spiked, t_j):
        """
        self.last_spike_time = t_i
        """ 
        dw = np.zeros((n,1))
        if self.last_spike_time > 0:
            t= t_j - self.last_spike_time
            ltd_val = self.ltd(delta_t= t)
            dw = first_time_spiked * ltd_val
        return dw
    
    def ltp(self, delta_t):
        return self.A_pos * np.exp(delta_t/self.tau_pos)

    def ltd(self, delta_t):
        return self.A_neg * np.exp(-delta_t/self.tau_neg)

    def eps_kernel(self, s):
        """"
        s = t - t_j 
        time difference between thre current time and spike time of presynaptic neuron (t_j)
        K is chosen such that the maximum value of epsilon kernel will be 1, based on the tau_m and tau_s.
        """
        s_max = (self.tau_m * self.tau_s) / (self.tau_s - self.tau_m) * np.log(self.tau_s / self.tau_m)
        max_val = (np.exp(-s_max/self.tau_m) - np.exp(-s_max/self.tau_s))
        K = 1 / max_val
        return K * (np.exp(-s/self.tau_m) - np.exp(-s/self.tau_s))
    
    def eta_kernel(self, s):
        positive_pulse = self.K1 * np.exp(-s/self.tau_m)
        negative_spike_afterpotential = self.K2 * (np.exp(-s/self.tau_m) - np.exp(-s/self.tau_s))
        return self.threshold * (positive_pulse - negative_spike_afterpotential)
    
    def create_epsilon_matrix(self, spike_train):
        matrix = np.zeros(spike_train.shape)
        if spike_train.ndim ==1:
            size = len(spike_train)
            for it in np.arange(size):
                matrix[it] = self.eps_kernel((size-it)*self.dt)
        else:
            size = spike_train.shape[1]
            for it in np.arange(size):
                matrix[:, it] = self.eps_kernel((size-it)*self.dt)
        return matrix
    
    def reset_neuron(self):
        self.last_spike_time = - 10e6
        self.spike_rec = typed.List.empty_list(np.float64)
        self.potential_rec = typed.List([0.0])
        # self.spike_rec = []
        # self.potential_rec = [0.0]
        self.ref_counter = 0
        self.afferents_not_spiked_yet = None
        self.w_sample = None
    
    def get_potential(self, spike_train_window, weight, t):
        epsilon_matrix = self.create_epsilon_matrix(spike_train_window)
        eps = spike_train_window * epsilon_matrix
        eps = np.sum(eps, axis=1).reshape((-1,1))
        eps = eps * weight
        eps = np.sum(eps)
        eta = self.eta_kernel(t+self.dt-self.last_spike_time)
        potential = eta + eps
        return potential

    def get_spike_train_window(self, index, spike_train):
        start_widnow = max(0, int((self.last_spike_time)/self.dt))
        ### throw away early spikes if (t-t_i) or (t-t_j) greater than 7*tau_m
        start_widnow = max(start_widnow, int(index - 7*self.tau_m/self.dt))
        end_window = index + 1
        spike_train_window = spike_train[:, start_widnow:end_window]

        return spike_train_window
    
    def get_stdp_window(self, index, spike_train):
        start_widnow = max(0, int((self.last_spike_time+self.dt)/self.dt))
        ### throw away early spikes if (t_j-t_i) greater than 7*tau_pos
        start_widnow = max(start_widnow, int(index - 7*self.tau_pos/self.dt))
        end_window = index
        stdp_window = spike_train[:, start_widnow:end_window]
        return stdp_window

    def run(self, spike_train, weight, pattern_times):
        n, total_length = spike_train.shape
        if self.w_sample is None:
            if len(weight) > 10:
                self.w_sample = np.ones((10,total_length)) * weight[:10]
            else:
                self.w_sample = np.ones((weight.shape[0],total_length)) * weight[:10]
        if self.afferents_not_spiked_yet is None:
            self.afferents_not_spiked_yet = np.ones(weight.shape, dtype=np.float64)
        if pattern_times:
            latency = typed.List.empty_list(np.float64)
            # latency = []
        else:
            latency = None
            
        for it in range(total_length): 
            t = it*self.dt
            # update pattern_times list
            if pattern_times:
                if t > pattern_times[0] + 0.05:
                    pattern_times.pop(0)
            
            if self.ref_counter > 0:
                self.ref_counter -= 1
            elif self.potential_rec and self.potential_rec[-1] > self.threshold:
                ##### STDP LTP Rule #####
                stdp_window = self.get_stdp_window(index=it, spike_train=spike_train)
                dw = self.do_ltp(stdp_window)
                weight += dw
                weight = np.clip(weight, 0, 1)
                # we spike
                self.last_spike_time = t
                self.spike_rec.append(t)
                self.ref_counter = (self.tref)/self.dt
                self.potential_rec[-1] = self.eta_kernel(t-self.last_spike_time)
                self.afferents_not_spiked_yet = np.ones(weight.shape, dtype=np.float64)
                ## 
                if pattern_times:
                    if t > pattern_times[0]:
                        latency.append(t - pattern_times[0])
                    
                
            spike_train_window = self.get_spike_train_window(index=it, spike_train=spike_train)

            if t > self.last_spike_time:
                afferents_spiked_now = spike_train_window[:, -1:]
            else:
                afferents_spiked_now = np.zeros(weight.shape, dtype=np.float64)

            potential = self.get_potential(spike_train_window, weight, t)
            self.potential_rec.append(potential)

            # ##### STDP LTD Rule #####
            first_time_spiked = afferents_spiked_now * self.afferents_not_spiked_yet
            dw = self.do_ltd(n, first_time_spiked, t)
            weight += dw
            weight = np.clip(weight, 0, 1)
            self.afferents_not_spiked_yet = self.afferents_not_spiked_yet - first_time_spiked

            self.w_sample[:,it:it+1] = weight[:10,:]
        
        return weight, latency