import sys
import yaml
from datetime import datetime
import random
import copy
import numpy

import torch
from torch.utils.data import DataLoader
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence

from modules import (
    obs_encoder,
    pi_encoder,
    r_aggregator,
    r_to_z,
    z_to_L0,
    conditional_ODE_func,
    decoder,
)
from models import LUPI_NDP, NDP
from generators import OscillatorsGenerator, LotkaVolterraGenerator

import utils

# config
if len(sys.argv) == 2:
    config_path = sys.argv[1]
elif len(sys.argv) > 2:
    raise (RuntimeError(("Wrong arguments, use" "python main.py <path_to_config>")))
else:
    config_path = "configs/default.yml"
print(f"Using config_path {config_path}.")
config = yaml.safe_load(open(config_path))

print(f"Config: {config}")

# ------ Seeding
if 'seed' in config.keys():
    torch.manual_seed(config['seed'])
    random.seed(config['seed'])
    numpy.random.seed(config['seed'])

# ------ Loading saved models
if config["use_saved_models"]:
    models = {
        model: torch.load("../data/saved_models/" + model)
        for model in config["saved_models"]
    }
else:
    models = {}
    if "lupi" in config["models"]:
        models["lupi"] = LUPI_NDP(config)
    if "nopi" in config["models"]:
        models["nopi"] = NDP(config)
    if not ("lupi" in config["models"] or "nopi" in config["models"]):
        raise ValueError(
            (
                "Not using saved models and models specified for training are invalid. ",
                'Try adding {models: ["lupi","nopi"]} to the config.',
            )
        )

# ------ Test-time metric tracking
stats_dict = {}
if config["mse"]:
    stats_dict["error"] = []
if config["calibration"]:
    # record the calibration score per task
    stats_dict["calibration_score"] = []
    # record the observed confidences per task, for an overall calibration
    stats_dict["confidences"] = numpy.zeros(config["calibration_bins"])
if config["sharpness"]:
    stats_dict["sharpness"] = []
if not (config["mse"] or config["calibration"] or config["sharpness"]):
    raise ValueError(
        "No valid evaluation metrics specified. Try adding {mse : true} to the config."
    )

tracked_stats = {
    key: {
        "train_mode": copy.deepcopy(stats_dict),
        "eval_mode": copy.deepcopy(stats_dict),
    }
    for key in models
}

# ------ Load or generate data
if config["use_saved_data"]:
    train_set = torch.load("../data/datasets/" + config["saved_train"])
    val_set = torch.load("../data/datasets/" + config["saved_val"])
    eval_set = torch.load("../data/datasets/" + config["saved_eval"])
else:
    if not ("task_type" in config.keys()):
        raise ValueError(
            (
                "Not using saved data and task type not given. "
                "Try adding {task_type : oscillators} to the config."
            )
        )

    if config["task_type"] == "oscillators":
        train_val_set = OscillatorsGenerator(config, train=True)
        train_set = train_val_set[
            : int(config["train_examples"] * config["train_val_split"])
        ]
        val_set = train_val_set[
            int(config["train_examples"] * config["train_val_split"]) :
        ]
        eval_set = OscillatorsGenerator(config, train=False)
    elif config["task_type"] == "lotka_volterra":
        train_val_set = LotkaVolterraGenerator(config, train=True)
        train_set = train_val_set[
            : int(config["train_examples"] * config["train_val_split"])
        ]
        val_set = train_val_set[
            int(config["train_examples"] * config["train_val_split"]) :
        ]
        eval_set = LotkaVolterraGenerator(config, train=False)
    else:
        raise ValueError(f"Task type {config['task_type']} not recognised.")

train_loader = DataLoader(train_set, batch_size=1)
val_loader = DataLoader(val_set, batch_size=1)
eval_loader = DataLoader(eval_set, batch_size=1)

# context and target ranges are used during training and evaluation
context_range = range(config["context_range"][0], config["context_range"][1])
target_range = range(config["target_range"][0], config["target_range"][1])

# ------ Train models (if not using saved)
if not config["use_saved_models"]:
    for key in models:
        if config["verbose"]:
            print(f"Training {key} model.")
        optimizer = torch.optim.Adam(models[key].parameters(), lr=config["lr"])
        for epoch in range(config["epochs"]):
            total_epoch_loss = 0.0
            recon_epoch_loss = 0.0
            kl_epoch_loss = 0.0
            for i, batch in enumerate(train_loader):
                optimizer.zero_grad()
                pi, x, times = batch
                # get context and targets
                context_size = random.sample(context_range, 1)[0]
                target_size = random.sample(target_range, 1)[0]
                # during training, context are a subset of targets
                target_idx = random.sample(
                    range(config["simulator"]["samples"]), target_size
                )
                context_idx = random.sample(target_idx, context_size)
                target_idx.sort()
                context_idx.sort()
                target_idx, context_idx = torch.LongTensor(
                    target_idx
                ), torch.LongTensor(context_idx)

                zC, zT, preds = models[key](
                    x.squeeze(0),
                    times.squeeze(0),
                    pi.squeeze(0),
                    context_idx,
                    target_idx,
                )

                # calculate loss
                log_likelihood = preds.log_prob(x.squeeze(0)[target_idx]).sum()
                kl = kl_divergence(zT, zC).sum()
                loss = -log_likelihood + kl
                # backprop
                loss.backward()
                optimizer.step()
                # record
                total_epoch_loss += loss.item()
                recon_epoch_loss += -log_likelihood.item()
                kl_epoch_loss += kl.item()

            if (epoch + 1) % config["val_and_print"] == 0:
                with torch.no_grad():
                    models[key].eval()
                    val_recon = 0

                    for i, batch in enumerate(val_loader):
                        pi, x, times = batch

                        # context/target idx
                        context_size = random.sample(context_range, 1)[0]
                        # during eval, context and targets are disjoint and targets are all but context
                        context_idx = random.sample(
                            range(config["simulator"]["samples"]), context_size
                        )
                        target_idx = list(
                            set(range(config["simulator"]["samples"]))
                            - set(context_idx)
                        )
                        context_idx.sort()
                        target_idx = torch.LongTensor(target_idx)
                        context_idx = torch.LongTensor(context_idx)
                        preds = models[key](
                            x.squeeze(0),
                            times.squeeze(0),
                            pi.squeeze(0),
                            context_idx,
                            target_idx,
                        )
                        log_likelihood = preds.log_prob(x.squeeze(0)[target_idx]).mean()

                        val_recon += -log_likelihood.item()
                if config["verbose"]:
                    print_string = (
                        "Epoch {0:d} \t"
                        "Train {1:.0f} / {2:.0f} / {3:.0f} \t"
                        "Val {4:.0f}"
                    ).format(
                        epoch + 1,
                        total_epoch_loss / len(train_loader),
                        recon_epoch_loss / len(train_loader),
                        kl_epoch_loss / len(train_loader),
                        val_recon / len(val_loader),
                    )
                    print(print_string)
                models[key].train()

if config["verbose"]:
    print("Starting evaluation.")

# evaluation (paired if using both models)
with torch.no_grad():
    for i, batch in enumerate(eval_loader):
        pi, x, times = batch
        # select context and target indices
        context_size = random.sample(context_range, 1)[
            0
        ]  # some number in the range 5 to 9
        target_size = random.sample(target_range, 1)[
            0
        ]  # some number in the range 15 to 50
        target_idx = list(range(config["simulator"]["samples"]))  # a set of indices
        context_idx = random.sample(
            target_idx, context_size
        )  # a subset of the target indices
        target_idx.sort()
        context_idx.sort()
        target_idx = torch.LongTensor(target_idx)
        context_idx = torch.LongTensor(context_idx)

        # sample predictions from each model and get mean and std
        pred = {key: {"train_mode": {}, "eval_mode": {}} for key in models}
        for key in models:
            train_mode = []
            eval_mode = []
            for _ in range(config["z_samples"]):
                # train mode: pi, full target for context
                models[key].train()
                _, _, train_pred = models[key](
                    x.squeeze(0),
                    times.squeeze(0),
                    pi.squeeze(0),
                    context_idx,
                    target_idx,
                )
                train_mode.append(train_pred.loc.detach())
                # eval mode: only context is used
                models[key].eval()
                eval_pred = models[key](
                    x.squeeze(0),
                    times.squeeze(0),
                    pi.squeeze(0),
                    context_idx,
                    target_idx,
                )
                eval_mode.append(eval_pred.loc.detach())
            train_mode = torch.stack(train_mode, 0)
            eval_mode = torch.stack(eval_mode, 0)
            (
                pred[key]["train_mode"]["mean"],
                pred[key]["train_mode"]["std"],
            ) = train_mode.mean(0), train_mode.std(0)
            (
                pred[key]["eval_mode"]["mean"],
                pred[key]["eval_mode"]["std"],
            ) = eval_mode.mean(0), eval_mode.std(0)

        xs = x.squeeze(0)[:-1]
        # track required statistics
        for key in models:
            # tm = train mode, em = eval mode
            tm_mean, tm_std = (
                pred[key]["train_mode"]["mean"],
                pred[key]["train_mode"]["std"],
            )
            em_mean, em_std = (
                pred[key]["eval_mode"]["mean"],
                pred[key]["eval_mode"]["std"],
            )
            if config["mse"]:
                tracked_stats[key]["train_mode"]["error"].append(
                    torch.mean((tm_mean - xs) ** 2).item()
                )
                tracked_stats[key]["eval_mode"]["error"].append(
                    torch.mean((em_mean - xs) ** 2).item()
                )
            if config["calibration"]:
                try:
                    t_conf = Normal(tm_mean, tm_std).cdf(xs).numpy()
                    e_conf = Normal(em_mean, em_std).cdf(xs).numpy()
                except: # low z-samples can be degenerate in prediction space
                    t_conf = Normal(tm_mean, tm_std+1e-9).cdf(xs).numpy()
                    e_conf = Normal(em_mean, em_std+1e-9).cdf(xs).numpy()
                # confidence to normalised cumulative histograms (= cumulative relative frequency)
                t_values, t_edges = numpy.histogram(
                    t_conf, config["calibration_bins"], range=(0.0, 1.0)
                )
                e_values, e_edges = numpy.histogram(
                    e_conf, config["calibration_bins"], range=(0.0, 1.0)
                )
                t_normcum = numpy.cumsum(t_values) / t_conf.size
                e_normcum = numpy.cumsum(e_values) / e_conf.size
                # cumulative histograms to scores
                midpoints = (t_edges[:-1] + t_edges[1:]) / 2.0
                t_score = ((t_normcum - midpoints) ** 2).sum()
                e_score = ((e_normcum - midpoints) ** 2).sum()
                # record scores
                tracked_stats[key]["train_mode"]["calibration_score"].append(t_score)
                tracked_stats[key]["eval_mode"]["calibration_score"].append(e_score)
                # add frequencies
                tracked_stats[key]["train_mode"]["confidences"] += t_normcum
                tracked_stats[key]["eval_mode"]["confidences"] += e_normcum
            if config["sharpness"]:
                tracked_stats[key]["train_mode"]["sharpness"].append(
                    torch.mean(tm_std).item()
                )
                tracked_stats[key]["eval_mode"]["sharpness"].append(
                    torch.mean(em_std).item()
                )

if config["verbose"]:
    print("Evaluation complete, calculating metrics")

# evaluation statistics
eval_metrics = utils.metrics(tracked_stats, config)
# append config for results
eval_metrics["config"] = config

save_string = datetime.today().strftime("../data/results/run_%Y_%m_%d_%H%M.yml")
with open(save_string, "w") as outfile:
    yaml.dump(eval_metrics, outfile)
print("Saved results to {" + save_string + "}.")
