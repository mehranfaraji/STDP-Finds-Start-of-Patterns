import numpy as np 

from typing import Iterable, Optional
import numpy

DEFAULT_DT = 0.001
dt = DEFAULT_DT

# pretty print matrix A
def pprint(A, precision=3):
    A = np.around(A, decimals=precision)
    if A.ndim==1:
        print(A)
    else:
        w = max([len(str(s)) for s in A]) 
        print(u'\u250c'+u'\u2500'*w+u'\u2510') 
        for AA in A:
            print(' ', end='')
            print('[', end='')
            for i,AAA in enumerate(AA[:-1]):
                w1=max([len(str(s)) for s in A[:,i]])
                print(str(AAA) +' '*(w1-len(str(AAA))+1),end='') # str(AAA)
            w1=max([len(str(s)) for s in A[:,-1]])
            print(str(AA[-1])+' '*(w1-len(str(AA[-1]))),end='') # str(AA[-1])
            print(']')
        print(u'\u2514'+u'\u2500'*w+u'\u2518')  
        

class SRM():
    def __init__(
            self, 
            N: int = None,
            threshold: float = None,
            reset: float = None,
            refractory: float = None, # refractory is a float time [second]
            tau_m: float = None,
            tau_s: float = None,
            K1: float = None,
            K2: float = None,
            window_time: float = None, # Maximum time [second] to ignore inputs before it by kernels 
            train_mode: bool = True
        ) -> None:
        """
        """
        assert N is not None, "Number of neurons must be provided." # TODO: write better check for the class 

        self.N = N
        self.threshold = threshold
        self.reset = reset
        self.refractory = refractory
        self.ref_counter = np.zeros((self.N,1))
        self.tau_m = tau_m
        self.tau_s = tau_s
        self.K1 = K1
        self.K2 = K2
        self.window_time = window_time
        self.train_mode = train_mode
        ## TODO: Do I need to tell the model about dt? dt should be given by user in net = Networ()
        self.dt = DEFAULT_DT
        ## TODO: I think we need to keep track of current time (t).
        ## TODO: Updating self.t will occur inside the net = Network? 
        # self.t = None
        self.last_spike_time = - np.ones((self.N,1)) * np.inf

        self.potential_rec = None
        self.monitor = None
    
            
    def forward(self, spikes_t: numpy.array, w_tmp: numpy.array, current_t: float, current_it, i: int):
        """
        spikes_t here is a correct window of spikes_t.
        tmp_w is the corresponding weights of spikes_t.
        current_t is ms 
        """
        # There's an input
        # TODO: maybe I can check if spikes_t is not empty before calling forward()
        if len(spikes_t) != 0:

            s = current_t + self.dt - spikes_t
            eps = self.eps_kernel(s)
            eps = eps * w_tmp
            # eps = eps.reshape((self.N,-1))
            # print(s)
            # pprint(w_tmp)
            # pprint(eps)
            # print(eps.shape)
            # eps = eps.sum(axis=1, keepdims=True)
            eps = eps.sum()
            s = current_t + self.dt - self.last_spike_time[i]
            eta = self.eta_kernel(s)
            # TODO: What to do with calculated pot.?
            potential = eta + eps
            # print(f"potential = ")
            # pprint(potential)
            self.potential_rec[i, current_it + 1] = potential.squeeze() # potential.squeeze()
        
        # no incoming input
        else:
            # pst = 0 since we dont have any incoming input train
            eps = 0
            s = current_t + self.dt - self.last_spike_time[i]
            eta = self.eta_kernel(s)
            # TODO: What to do with calculated pot.?
            potential = eta + eps
            self.potential_rec[i, current_it + 1] = potential.squeeze()


        return potential

    def get_potential_i(self,spikes_t: numpy.array, w_tmp: numpy.array, current_t: float, current_it: float, i: int):
        
        potential = self.forward(spikes_t, w_tmp, current_t, current_it, i)

        return potential
    
    
    # def spike_check(self, potential: float):
    #     if potential >= self.threshold:
    #         return True
    #     else:
    #         return False
        
    def check_refractory(self):
        idx = self.ref_counter > 0
        self.ref_counter[idx] -= 1
        self.ref_counter[idx] = np.round(self.ref_counter[idx], 0) 
        return ~idx  

    def start_refractory(self,idx, current_t):
        self.ref_counter[idx] = self.refractory / self.dt
        self.last_spike_time[idx] = current_t
    
    
    def reset_neuron(self,):
        ### what should be reseted exactly?
        self.t = 0
        self.last_spike_time = - np.inf

    def init_records(self, time):
        T = time + 1
        self.potential_rec = np.zeros((self.N, T))
        # if self.monitor:
        #     self.monitor.init_records(time)
    
    
    ## TODO: use compute_k() when adding the layer into the Network()
    def compute_K(self):
        """
        K is chosen such that the maximum value of epsilon kernel will be 1, based on the tau_m and tau_s.
        """
        s_max = (self.tau_m * self.tau_s) / (self.tau_s - self.tau_m) * np.log(self.tau_s / self.tau_m)
        max_val = (np.exp(-s_max/self.tau_m) - np.exp(-s_max/self.tau_s))
        self.K = 1 / max_val
            
    def eps_kernel(self, s: numpy.array):
        """"
        s = t - t_j 
        time difference between thre current time and spike time of presynaptic neuron (t_j)
        """
        if not hasattr(self, 'K'):
            self.compute_K()
        return self.K * (np.exp(-s/self.tau_m) - np.exp(-s/self.tau_s))

    def eta_kernel(self, s: numpy.array):
        """
        s = t - t_i
        """
        positive_pulse = self.K1 * np.exp(-s/self.tau_m)
        negative_spike_afterpotential = self.K2 * (np.exp(-s/self.tau_m) - np.exp(-s/self.tau_s))
        return self.threshold * (positive_pulse - negative_spike_afterpotential)


class InputTrain():
    def __init__(self,
                spikes_t,
                spikes_i) -> None:
        self.spikes_t = spikes_t
        self.spikes_i = spikes_i

class Monitor():
    def __init__(self,
                 layer: SRM) -> None:
        layer.monitor = self
        ## TODO: Is there a way to know how many times each of the postsynaptic neurons will spike?
        self.spikes_t = np.array([])
        self.spikes_i = np.array([])
        ## TODO: Maybe also mode potential_rec to monitor object instead of SRM model itself.
        # self.potential_rec = None
        self.dt = DEFAULT_DT

    def record_spike(self, current_t, idx):
        tmp_t = [current_t] * (idx)
        tmp_t = tmp_t[idx]
        self.spikes_t = np.append(self.spikes_t, tmp_t)
        tmp_i = np.where(idx)[0]
        self.spikes_i = np.append(self.spikes_i, tmp_i)        


class Synapse():
    def __init__(
                self,
                w: numpy.array,
                w_max: float,
                w_min: float,
                A_pre: float,
                A_post: float,
                tau_pre: float,
                tau_post: float,
                approximate: bool = False
                ) -> None:
        self.w = w
        self.w_max = w_max
        self.w_min = w_min
        self.A_pre = A_pre
        self.A_post = A_post
        self.tau_pre = tau_pre
        self.tau_post = tau_post
        ## if True do nearest spike approximation, else consider all the contributions of the previous presynaptic spikes
        self.approximate = approximate
        
        self.a_pre = np.zeros_like(self.w)
        self.a_post = np.zeros_like(self.w)
        self.dt = DEFAULT_DT
        # self.afferents_spiked = np.zeros((self.w.shape[0],1))

        if self.A_pre < 0:
            raise ValueError("A_pre should be > 0")
        if self.A_post > 0:
            raise ValueError("A_post should be < 0")
        if len(self.w.shape) < 2:
            raise ValueError("w should be 2D")
    
    def get_w_tmp(self, spikes_i: numpy.array, i):
        """
        spikes_i contains only the correct window of spikes_i
        """
        return self.w[spikes_i, i].squeeze().T

    def on_pre(self,idx):
        """
        idx is the index of presynaptic neurons that spiked at the current time and now increasing their a_pre,
        idx is a python list
        """
        if self.approximate:
            self.a_pre[idx, :] = self.A_pre
            # self.a_pre[self.afferents_spiked] += self.A_pre
        else: self.a_pre[idx, :] += self.A_pre
              # self.a_pre[self.afferents_spiked] += self.A_pre
        self.w[idx, :] = np.clip(self.w[idx, :] + self.a_post[idx, :], self.w_min, self.w_max)
        # self.w = np.clip(self.w + self.a_post[self.afferents_spiked], self.w_min, self.w_max)

    def on_post(self, idx):
        """
        idx is the index of postsynaptic neurons not in refractory period and their membrane potential above threshold
        """
        if self.approximate:
            self.a_post[:, idx.squeeze()] = self.A_post
        else: self.a_post[:, idx.squeeze()] += self.A_post
        ## TODO: Only the weights of the postsynaptic neurons that are spiking at the current time should be updated
        self.w[:, idx.squeeze()] = np.clip(self.w[:, idx.squeeze()] + self.a_pre[:, idx.squeeze()], self.w_min, self.w_max)

    def update_a(self):
        self.a_pre = self.a_pre - self.dt / self.tau_pre * self.a_pre
        self.a_post = self.a_post - self.dt / self.tau_post * self.a_post


class NetworkBase():
    def __init__(
            self,
            ) -> None:
        self.synapses = []
        self.layers = []
        self.potential = []
        self.input_train = None
        self.dt = DEFAULT_DT
        self.current_t = 0.0 ## also add in reset()!

    def add_synapse(self, synapse: Synapse):
        self.synapses.append(synapse)
        

    def add_layer(self, layer: SRM):
        self.layers.append(layer)
        self.potential.append(np.zeros((layer.N, 1)))
        
    def add_input_train(self, input_train: InputTrain):
        self.input_train = input_train

    def init_records(self,time):
        # TODO: remove the line below in the future
        # so that we can have multiple net.run() and it considers
        # current time of previous runs!
        self.current_t = 0.0
        self.current_it = 0
        for layer in self.layers:
            layer.init_records(time)


    def check_post_spike(self, layer: SRM, synapse: Synapse, monitor: Monitor, idx: numpy.array):
        """
        idx is the index of the neurons not in refractory period and their membrane potential above threshold
        current_idx = int(current_t/ dt)
        """
    
        synapse.on_post(idx)
        if monitor:
            monitor.record_spike(self.current_t, idx)
        layer.start_refractory(idx, self.current_t)
        layer.potential_rec[idx.squeeze(), self.current_it] = layer.eta_kernel(0.0)
        # TODO: I think line below is unnecessary! as it gets updated each iteration
        # self.potential[idx_spike] = layer.eta_kernel(0.0)
        
        # synapse.afferents_spiked = np.zeros((synapse.w.shape[0],1))


class Network(NetworkBase):
    def __init__(self,):
        super().__init__()
    
    def get_potential(self, layer, l):
        spikes_t = self.input_train.spikes_t
        spikes_i = self.input_train.spikes_i
        potential = [] 
        for i in range(layer.N):
            
            start_idx = max(0, self.current_t - 7 * layer.tau_m, layer.last_spike_time[i].squeeze())
            start_slice = np.searchsorted(spikes_t, start_idx, side='left')
            end_slice = np.searchsorted(spikes_t, self.current_t, side='right')
            # update index value
            end_slice = end_slice
            spikes_t_i = spikes_t[start_slice:end_slice]
            spikes_i_i = spikes_i[start_slice:end_slice]

            w_tmp = self.synapses[l].get_w_tmp(spikes_i_i, i)
            potential_i = layer.get_potential_i(spikes_t_i, w_tmp, self.current_t, self.current_it, i)
            potential.append(potential_i)
        self.potential = np.array(potential)

    
    def run_one_step(self):
        spikes_t = self.input_train.spikes_t
        spikes_i = self.input_train.spikes_i
        for l, layer in enumerate(self.layers):
            # idx : index of neurons not in refractory period.
            idx = layer.check_refractory()
            idx_spike = self.potential[l] >= layer.threshold
            idx_spike = idx * idx_spike
            self.check_post_spike(layer=layer, synapse=self.synapses[l], monitor=layer.monitor, idx=idx_spike)
            ## WARNING:
            ## TODO: There is a bug here when a deep network is defined.
            ## here istead of a usinng spike_t which is from the InputTrain
            ## we should instead use the spike train from previous layer not the input!
            
            self.get_potential(layer, l)


            # STDP LTD Rule
            # print(spikes_t)
            idx_right = np.searchsorted(spikes_t, self.current_t, side='right')
            idx_left = np.searchsorted(spikes_t, self.current_t, side='left')
            # print(idx_left, idx_right)
            # pprint(spikes_i[idx_left:idx_right])
            # print(len(spikes_i[idx_left:idx_right]), len(set(spikes_i[idx_left:idx_right])))

            self.synapses[l].on_pre(spikes_i[idx_left:idx_right])
        
            

    def run(self, time: float):
        """
        time (ms)
        """
        self.init_records(time)
        ## Check the behaviour of + self.dt on the last loop step
        for it in range(time):
            self.current_it = it
            # print(f"{it = }\n")
            self.run_one_step()


            self.current_it = it
            self.current_t += self.dt
            self.current_t = np.round(self.current_t,3)

            if it % 20000 == 0:
                print(f"{it = }")




