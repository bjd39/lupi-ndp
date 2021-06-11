import torch
import warnings

class obs_encoder(torch.nn.Module):
    """
    Takes raw observations (x_i, t_i) and produces a representation r_i.
    """
    
    def __init__(self,config):
        super(obs_encoder, self).__init__()
        
        obs_dim = config['x_dim']
        h_dim = config['obs_enc_h_dim']
        r_dim = config['r_dim']
        
        layers = [
            torch.nn.Linear(obs_dim+1,h_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(h_dim,h_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(h_dim,r_dim)
        ]
        
        self.layers = torch.nn.Sequential(*layers)
    
    def forward(self, observations, times):
        return self.layers(torch.cat([observations, times.unsqueeze(-1)],-1))
    
class pi_encoder(torch.nn.Module):
    """
    Takes the privileged information signal and produces a representation r_p.
    """
    def __init__(self, config):
        super(pi_encoder, self).__init__()
        
        pi_dim = config['pi_dim']
        h_dim = config['pi_enc_h_dim']
        r_dim = config['pi_r_dim']
        
        layers = [
            torch.nn.Linear(pi_dim,h_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(h_dim,h_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(h_dim,r_dim)
        ]
        
        self.layers = torch.nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)

class r_aggregator(torch.nn.Module):
    """
    Takes mixed representations from observations and privileged information and
    produces a single task representation.
    
    In evaluation mode privileged information should not be used and a warning will
    given if this is attempted.
    """
    def __init__(self, config):
        super(r_aggregator, self).__init__()
        
        obs_r_dim = config['r_dim']
        pi_r_dim = config['pi_r_dim']
        h_dim = config['agg_h_dim']
        
        self.aggregators = config['aggregators']
        
        for agg in self.aggregators:
            if agg not in ['mean','max','min','logsumexp']:
                raise ValueError(f'Aggregator {agg} is not supported.')
        
        # when using privileged information we combine the representations with a small ResNet
        layers = [
            torch.nn.Linear(obs_r_dim*len(self.aggregators)+pi_r_dim,h_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(h_dim,h_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(h_dim,obs_r_dim*len(self.aggregators))
        ]
        
        self.layers = torch.nn.Sequential(*layers)
        
    def aggregate(self, obs_r):
        # concatentate the aggregator outputs
        agg = []
        if 'max' in self.aggregators:
            agg.append(torch.max(obs_r,0).values)
        if 'min' in self.aggregators:
            agg.append(torch.min(obs_r,0).values)
        if 'logsumexp' in self.aggregators:
            agg.append(torch.logsumexp(obs_r,0))
        if 'mean' in self.aggregators:
            agg.append(torch.mean(obs_r,0))
        
        return torch.cat(agg, 0)
        
    def forward(self, r_i, r_p=None):
        if not self.training and r_p is not None:
            warnings.warn('Privileged information provided to the aggregator in evaluation mode.')
        
        # aggregate observations
        r_obs = self.aggregate(r_i)
        
        # if we have privileged information, use the resnet
        # r_obs' = r_obs + f(r_obs, r_p)
        if r_p is not None:
            r_obs = r_obs + self.layers(torch.cat([r_obs, r_p], -1))
        
        return r_obs

class r_to_z(torch.nn.Module):
    """
    From task representation to global latent variable.
    """
    def __init__(self, config):
        super(r_to_z, self).__init__()
        
        aggregated_r_dim = config['r_dim'] * len(config['aggregators'])
        h_dim = config['r_to_z_h_dim']
        z_dim = config['z_dim']
        
        layers = [
            torch.nn.Linear(aggregated_r_dim,h_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(h_dim,h_dim),
            torch.nn.ReLU()
        ]
        
        self.layers = torch.nn.Sequential(*layers)
        
        self.h_to_mu = torch.nn.Linear(h_dim, z_dim)
        self.h_to_sigma = torch.nn.Linear(h_dim, z_dim)
        
    def forward(self, r):
        h = self.layers(r)
        mu = self.h_to_mu(h)
        sigma = 0.1 + 0.9*torch.sigmoid(self.h_to_sigma(h))
        
        return torch.distributions.Normal(mu, sigma)

class z_to_L0(torch.nn.Module):
    """
    Use a sample from the global latent variable to produce the initial L value.
    """
    def __init__(self,config):
        super(z_to_L0, self).__init__()
        
        L_dim = config['L_dim']
        z_dim = config['z_dim']
        h_dim = config['z_to_L_h_dim']
        
        layers = [
            torch.nn.Linear(z_dim,h_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(h_dim,h_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(h_dim,L_dim)
        ]
        
        self.layers = torch.nn.Sequential(*layers)
        
    def forward(self, z_sample):
        return self.layers(z_sample)

class conditional_ODE_func(torch.nn.Module):
    """
    Conditional derivative function.
    
    Input is a concatentation of the ODE latent state and the global latent variable sample,
    and the time
        (L || z, t)
    Output is the derivative at that time
        (dL || dz)
    with dz set to 0.
    
    This is a 'hacky' way to condition the ODE without letting the sample varying.
    """
    
    def __init__(self,config):
        super(conditional_ODE_func, self).__init__()
        
        L_dim = config['L_dim']
        z_dim = config['z_dim']
        h_dim = config['ode_h_dim']
        
        layers = [
            torch.nn.Linear(L_dim+z_dim+1,h_dim),
            torch.nn.Softplus(),
            torch.nn.Linear(h_dim,h_dim),
            torch.nn.Softplus(),
            torch.nn.Linear(h_dim,L_dim)
        ]
        
        self.latent_func = torch.nn.Sequential(*layers)
        
        self.L_dim = L_dim
        
        self.nfe = 0
    
    def forward(self, t, L_z):
        self.nfe += 1
        z = L_z[self.L_dim:]
        L_z_t = torch.cat([L_z,t.unsqueeze(-1)])
        dL = self.latent_func(L_z_t)
        dz = torch.zeros_like(z)
        
        return torch.cat([dL,dz])

class decoder(torch.nn.Module):
    """
    Produce an output from the ODE latent state and the global latent variable sample.
    """
    def __init__(self, config):
        super(decoder, self).__init__()
        
        L_dim = config['L_dim']
        z_dim = config['z_dim']
        h_dim = config['dec_h_dim']
        out_dim = config['x_dim']
        
        layers = [
            torch.nn.Linear(L_dim+z_dim,h_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(h_dim,h_dim),
            torch.nn.ReLU()
        ]
        
        self.shared_layers = torch.nn.Sequential(*layers)
        
        self.hidden_to_mu = torch.nn.Linear(h_dim, out_dim)
        self.hidden_to_sigma = torch.nn.Linear(h_dim, out_dim)
        
    def forward(self, L_z_t):
        
        hidden = self.shared_layers(L_z_t)
        mu = self.hidden_to_mu(hidden)
        sigma = 0.1 + 0.9*torch.nn.functional.softplus(self.hidden_to_sigma(hidden))
        return torch.distributions.Normal(mu, sigma)