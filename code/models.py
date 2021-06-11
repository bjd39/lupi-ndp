import torch
from torchdiffeq import odeint

from modules import obs_encoder, pi_encoder, r_aggregator, r_to_z, z_to_L0, conditional_ODE_func, decoder

class LUPI_NDP(torch.nn.Module):
    
    def __init__(self, config):

        super(LUPI_NDP, self).__init__()
        
        self.observations_encoder = obs_encoder(config)
        self.privileged_info_encoder = pi_encoder(config)
        self.representation_aggregator = r_aggregator(config)
        self.representation_to_global_latent = r_to_z(config)
        self.global_latent_to_initial_value = z_to_L0(config)
        self.odefunc = conditional_ODE_func(config)
        self.decoder = decoder(config)
        
    def forward(self, observations, times, p_info, context_idx, target_idx):
        if self.training:
            # encode
            r_i = self.observations_encoder(observations, times)
            r_p = self.privileged_info_encoder(p_info)
            # aggregate
            r_C = self.representation_aggregator(r_i[context_idx],None)
            r_T = self.representation_aggregator(r_i[target_idx],r_p)
            # parametrise
            z_C = self.representation_to_global_latent(r_C)
            z_T = self.representation_to_global_latent(r_T)
            # sample (from target Z during training)
            z_ = z_T.rsample()
            # initialise and evolve
            L0 = self.global_latent_to_initial_value(z_)
            L_z = torch.cat([L0,z_])
            ################ this bit won't work with batching yet
            L_z_t = odeint(self.odefunc, L_z, times[target_idx])
            return z_C, z_T, self.decoder(L_z_t)
        
        else:
            # encode
            r_i = self.observations_encoder(observations[context_idx],
                                            times[context_idx])
            # aggregate
            r_C = self.representation_aggregator(r_i,None)
            # parametrise
            z_C = self.representation_to_global_latent(r_C)
            # sample (from context Z during eval.)
            z_ = z_C.rsample()
            # initialise and evolve
            L0 = self.global_latent_to_initial_value(z_)
            L_z = torch.cat([L0,z_])
            ################ this bit won't work with batching yet
            L_z_t = odeint(self.odefunc, L_z, times[target_idx].squeeze())
            return self.decoder(L_z_t)
        
class NDP(torch.nn.Module):
    """
    No privileged information NDP model built to sub in directly for the LUPI
    model for the sake of experimental consistency. Primarily this means that
    the forward pass takes a 'p_info' argument despite not using it.
    """
    
    def __init__(self, config):

        super(NDP, self).__init__()
        
        self.observations_encoder = obs_encoder(config)
        self.representation_aggregator = r_aggregator(config)
        self.representation_to_global_latent = r_to_z(config)
        self.global_latent_to_initial_value = z_to_L0(config)
        self.odefunc = conditional_ODE_func(config)
        self.decoder = decoder(config)
        
    def forward(self, observations, times, p_info, context_idx, target_idx):
        if self.training:
            # encode
            r_i = self.observations_encoder(observations, times)
            # aggregate (no PI so both aggregate without it)
            r_C = self.representation_aggregator(r_i[context_idx],None)
            r_T = self.representation_aggregator(r_i[target_idx],None)
            # parametrise
            z_C = self.representation_to_global_latent(r_C)
            z_T = self.representation_to_global_latent(r_T)
            # sample (from target Z during training)
            z_ = z_T.rsample()
            # initialise and evolve
            L0 = self.global_latent_to_initial_value(z_)
            L_z = torch.cat([L0,z_])
            ################ this bit won't work with batching yet
            L_z_t = odeint(self.odefunc, L_z, times[target_idx])
            return z_C, z_T, self.decoder(L_z_t)
        
        else:
            # encode
            r_i = self.observations_encoder(observations[context_idx],
                                            times[context_idx])
            # aggregate
            r_C = self.representation_aggregator(r_i,None)
            # parametrise
            z_C = self.representation_to_global_latent(r_C)
            # sample (from context Z during eval.)
            z_ = z_C.rsample()
            # initialise and evolve
            L0 = self.global_latent_to_initial_value(z_)
            L_z = torch.cat([L0,z_])
            ################ this bit won't work with batching yet
            L_z_t = odeint(self.odefunc, L_z, times[target_idx].squeeze())
            return self.decoder(L_z_t)