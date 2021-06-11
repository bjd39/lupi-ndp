import sys
import yaml

import torch
from torch.utils.data import DataLoader

from modules import obs_encoder, pi_encoder, r_aggregator, r_to_z, z_to_L0, conditional_ODE_func, decoder
from models import LUPI_NDP, NDP

# config
if len(sys.argv == 2):
    config_path = sys.argv[1]
elif len(sys.argv) > 2:
    raise(RuntimeError(("Wrong arguments, use"
                        "python main.py <path_to_config>")))
else:
    config_path = "configs/default.yml"
config = yaml.safe_load(open(config_path))

if not(config['mse'] or config['calibration'] or config['sharpness']):
    raise ValueError('No evaluation metrics specified. (Try adding {mse : true} to the config.)')

print(f"Config: {config}")

if config['use_saved_models']:
    models = [torch.load('../data/saved_models/'+model) for model in config['saved_models']]
else:
    if config["models"] == "both":
        models = [LUPI_NDP(config), NDP(config)]
        model_names = ['lupi','nopi']
    elif config["models"] == "nopi":
        models = [NDP(config)]
        model_names = ['nopi']
    elif config["models"] == "lupi":
        models = [LUPI_NDP(config)]
        model_names = ['lupi']
    else:
        print('Not using saved models and no model type specified: defaulting to LUPI-NDP')
        models = [LUPI_NDP(config)]
        model_names = ['lupi']

if config['use_saved_data']:
    train_set = torch.load('../data/datasets/'+config['saved_train'])
    val_set = torch.load('../data/datasets/'+config['saved_val'])
    eval_set = torch.load('../data/datasets/'+config['saved_eval'])

train_loader = DataLoader(train_set, batch_size=1)
val_loader = DataLoader(train_set, batch_size=1)
eval_loader = DataLoader(train_set, batch_size=1)

# train models if not using saved
if not config['use_saved_models']:
    # training loop

# evaluation (paired if using both models)
with torch.no_grad():
    if config['mse']:
        train_mode_errors = [[]]*len(models)
        eval_mode_errors = [[]]*len(models)
    if config['calibration']:
        train_mode_confidence = [[]]*len(models)
        eval_mode_confidence = [[]]*len(models)
    if config['sharpness']:
        train_mode_sharpness = [[]]*len(models)
        eval_mode_sharpness = [[]]*len(models)
    
    for (pi, x, times) in eval_loader:
        
        eval_mode_pred = [[]]*len(models)
        train_mode_pred = [[]]*len(models)
        
        # select context and target indices
        context_range = range(config['context_range'][0],config['context_range'][1])
        target_range = range(config['target_range'][0],config['target_range'][1])
        context_size = random.sample(context_range,1)[0] # some number in the range 5 to 9
        target_size = random.sample(target_range,1)[0] # some number in the range 15 to 50
        target_idx = list(range(config['simulator']['samples'])) # a set of indices
        context_idx = random.sample(target_idx, context_size) # a subset of the target indices
        target_idx.sort()
        context_idx.sort()
        target_idx = torch.LongTensor(target_idx)
        context_idx = torch.LongTensor(context_idx)
        
        for _ in range(z_samples):
            for model_idx, model in enumerate(models):
                # train mode: pi, full target for context
                model.train()
                _,_,train_pred = model(x.squeeze(0), times.squeeze(0), pi.squeeze(0), context_idx, target_idx)
                train_mode_pred[model_idx].append(train_pred.loc.detach())
                # eval mode: only context is used
                model.eval()
                eval_pred = model(x.squeeze(0), times.squeeze(0), pi.squeeze(0), context_idx, target_idx)
                eval_mode_pred[model_idx].append(eval_pred.loc.detach())
                
        # aggregate predictions over z-samples
        train_mean = [torch.stack(preds,0).mean(0) for preds in train_mode_pred]
        train_std = [torch.stack(preds,0).std(0) for preds in train_mode_pred]
        eval_mean = [torch.stack(preds,0).mean(0) for preds in eval_mode_pred]
        eval_std = [torch.stack(preds,0).std(0) for preds in eval_mode_pred]
        
        for idx,_ in enumerate(models):
            if config['mse']:
                train_mode_errors[idx].append(torch.mean((train_mean[idx] - x.squeeze(0)[:-1])**2).item())
                eval_mode_errors[idx].append(torch.mean((eval_mean[idx] - x.squeeze(0)[:-1])**2).item())
            if config['calibration']:
                train_mode_confidences[idx].append(Normal(train_mean[idx], train_std[idx]).cdf(x.squeeze(0)[:-1]))
                eval_mode_confidences[idx].append(Normal(eval_mean[idx], eval_std[idx]).cdf(x.squeeze(0)[:-1]))
            if config['sharpness']:
                train_mode_sharpness[idx].append(torch.mean(train_std[idx]).item())
                eval_mode_sharpness[idx].append(torch.mean(eval_std[idx]).item())
    
