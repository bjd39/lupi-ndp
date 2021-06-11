import torch
import numpy
from copy import deepcopy

class OscillatorsGenerator(torch.utils.data.Dataset):
    """
    Dampled Coupled Oscillators synthetic dataset.
    
    Produces data for the system
    
    |-spring-mass-spring-mass-spring-| w/ 'air' resistance
    
    where | are walls (i.e. immovable boundaries), described by
    
    f_i = k*(x_j - 2*x_i) - c*v_i
    
    for force f, displacements x, and velocity v, using Euler integration.
    
    Parameters
        n       : int, number of samples
        dt      : float, time step
        freq    : int, how frequently to sample (multiple of dt)
        samples : int, how many samples to take (multiples of freq)
                  (an extra sample is include for t=0)
    (possible) sampling parameters
        m1,m2   : masses
        x1,x2   : initial positions
        v1,v2   : initial velocities
        k       : (ideal) spring constant
        c       : 'air' resistance constant
    If a float is provided for a sampling parameter, it is fixed at that value;
    if a list is provided, it will be uniformly sampled from the range.

    The time step, freq and samples default to
        dt      : 0.01
        freq    : 10
        samples : 10
    n defaults to 1.
    
    The 'inputs' are the sampled parameters, i.e. if only the initial
    positions are left unspecified and the 'outputs' are the positions
    over time (x1(t),x2(t)).
    
    Output times are normalised to run 0 to 1, such that the first sample
    is always at time = 0 and the final is always at time = 1.
    """
    
    def __init__(self, config, train=True, verbose=True):

        # from config
        #   base simulator setup
        self.dt = config['simulator']['dt']
        self.freq = config['simulator']['sampling_freq']
        self.samples = config['simulator']['samples']
        #   system parameter dictionaries
        self.fixed = config['fixed']
        if train:
            self.varying = config['varying']
            self.n = config['train_examples']
        else:
            self.varying = config['eval_varying']
            self.n = config['eval_examples']
        
        # + 1 for the extra sample at t = 0
        self.total_steps = self.freq*self.samples + 1
        # normalised times
        self.times = torch.arange(0,self.samples+1,1)/(1.*self.samples)
        
        if verbose:
            if train:
                print('Training data generation...')
            else:
                print('Evaluation data generation...')
            print('Fixed paramters', self.fixed)
            print('Varying parameters', self.varying)
        
        self.data = []
        # generate data
        for _ in range(self.n):
            description, conditions = self.description_and_conditions()
            output = self.single_run(conditions)
            self.data.append((description, output, self.times))
        
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return self.n
    
    def description_and_conditions(self):
        """
        Return a sample of initial conditions, with those specified at
        input being fixed and the unspecified being the description, too.
        """
        
        conditions = deepcopy(self.fixed)
        for key in self.varying:
            if key in conditions:
                warnings.warn('System parameter defined in both fixed and varying configs')
            low = min(self.varying[key])
            high = max(self.varying[key])
            conditions[key] = numpy.random.uniform(low,high)
        
        # the description is the varying part
        description = [conditions[v] for v in self.varying]
        
        return torch.FloatTensor(description), conditions
    
    def single_run(self, conditions):
        x1, x2 = conditions['x1'], conditions['x2']
        v1, v2 = conditions['v1'], conditions['v2']
        m1, m2 = conditions['m1'], conditions['m2']
        c, k = conditions['c'], conditions['k']
        output = []
        for step in range(self.total_steps):
            if step % self.freq == 0:
                output.append([x1,x2])
            
            a1 = k*(x2-2*x1) - c*v1
            a2 = k*(x1-2*x2) - c*v2
            
            x1 += self.dt*v1
            x2 += self.dt*v2
            v1 += self.dt*a1
            v2 += self.dt*a2
    
        return torch.FloatTensor(output)
    
class LotkaVolterraGenerator(torch.utils.data.Dataset):
    """
    Generates a deterministic* Lotka-Voleterra synthetic dataset.
    
    (* population changes are not stochastic, data generation is random)
    
    Produces data for the system of predator vs prey
    
    Prey population is u, predator population is v, the dynamics
    are given by:
    
    u' = au - buv
    v' = duv - cv
    
    There is also a conserved quantity V (this is the privileged information):
    
    V = du - cln(u) + bv - aln(v)
    
    Parameters
        n       : int, number of samples
        dt      : float, time step
        freq    : int, how frequently to sample (multiple of dt)
        samples : int, how many samples to take (multiples of freq)
                  (an extra sample is include for t=0)
    (possible) sampling parameters
        u1,v1   : initial populations
        a       : reproduction rate
        b       : predation rate
        c       : death rate
        d       : predator reproduction rate
    If a float is provided for a sampling parameter, it is fixed at that value;
    if a list is provided, it will be uniformly sampled from the range.

    The time step, freq and samples default to
        dt      : 0.01
        freq    : 10
        samples : 10
    n defaults to 1.
    
    The 'inputs' are the sampled parameters, i.e. if only the initial
    positions are left unspecified and the 'outputs' are the positions
    over time (x1(t),x2(t)).
    
    Output times are normalised to run 0 to 1, such that the first sample
    is always at time = 0 and the final is always at time = 1.
    """
    
    def __init__(self, config, train=True, verbose=True):

        # from config
        #   base simulator setup
        self.dt = config['simulator']['dt']
        self.freq = config['simulator']['sampling_freq']
        self.samples = config['simulator']['samples']
        #   system parameter dictionaries
        self.fixed = config['fixed']
        if train:
            self.varying = config['varying']
            self.n = config['train_examples']
        else:
            self.varying = config['eval_varying']
            self.n = config['eval_examples']
        
        # + 1 for the extra sample at t = 0
        self.total_steps = self.freq*self.samples + 1
        # normalised times
        self.times = torch.arange(0,self.samples+1,1)/(1.*self.samples)
        
        if verbose:
            if train:
                print('Training data generation...')
            else:
                print('Evaluation data generation...')
            print('Fixed paramters', self.fixed)
            print('Varying parameters', self.varying)
        
        self.data = []
        # generate data
        for _ in range(self.n):
            description, conditions = self.description_and_conditions()
            output = self.single_run(conditions)
            self.data.append((description, output, self.times))
        
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return self.n
    
    def description_and_conditions(self):
        """
        Return a sample of initial conditions, with those specified at
        input being fixed and the unspecified being the description, too.
        """
        
        conditions = deepcopy(self.fixed)
        for key in self.varying:
            if key in conditions:
                warnings.warn('System parameter defined in both fixed and varying configs')
            low = min(self.varying[key])
            high = max(self.varying[key])
            conditions[key] = numpy.random.uniform(low,high)
        
        # the description is the varying part
        u1 = conditions['u1']
        v1 = conditions['v1']
        a = conditions['a']
        b = conditions['b']
        c = conditions['c']
        d = conditions['d']
        description = [d*u1 - c*numpy.log(u1) + b*v1 - a*numpy.log(v1)]
        
        return torch.FloatTensor(description), conditions
    
    def single_run(self, conditions):
        u = conditions['u1']
        v = conditions['v1']
        a = conditions['a']
        b = conditions['b']
        c = conditions['c']
        d = conditions['d']
        output = []
        for step in range(self.total_steps):
            if step % self.freq == 0:
                output.append([u,v])
            
            du = (a*u - b*u*v)*self.dt
            dv = (d*u*v - c*v)*self.dt
            
            u += du
            v += dv
    
        return torch.FloatTensor(output)