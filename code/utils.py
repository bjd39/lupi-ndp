import numpy
import copy


def metrics(tracked_stats, config):
    """
    Computes final evaluation metrics from the statistics tracked during evaluation.

    Converts outputs to regular floats (from numpy scalar floats) to make the yaml
    file human-legible (the loss of precision is negligible and << error.)
    """
    stats_dict = {}
    if config["mse"]:
        stats_dict["mse"] = {}
    if config["calibration"]:
        stats_dict["calibration_expected"] = {}
        stats_dict["calibration_overall"] = 0.0
    if config["sharpness"]:
        stats_dict["sharpness"] = {}
    eval_stats = {
        model: {
            "train_mode": copy.deepcopy(stats_dict),
            "eval_mode": copy.deepcopy(stats_dict),
        }
        for model in tracked_stats
    }

    for model in tracked_stats:
        for mode in ["train_mode", "eval_mode"]:
            if config["mse"]:
                error = numpy.array(tracked_stats[model][mode]["error"])
                mean, stderr = error.mean(), error.std() / error.size ** 0.5
                eval_stats[model][mode]["mse"]["mean"] = float(mean)
                eval_stats[model][mode]["mse"]["stderr"] = float(stderr)
            if config["calibration"]:
                # expected calibration
                score = numpy.array(tracked_stats[model][mode]["calibration_score"])
                mean, stderr = score.mean(), score.std() / score.size ** 0.5
                eval_stats[model][mode]["calibration_expected"]["mean"] = float(mean)
                eval_stats[model][mode]["calibration_expected"]["stderr"] = float(
                    stderr
                )
                # total calibration
                _, edges = numpy.histogram(
                    0, bins=config["calibration_bins"], range=(0.0, 1.0)
                )
                midpoints = (edges[:-1] + edges[1:]) / 2
                freq = tracked_stats[model][mode]["confidences"]
                # freq is sum of relative frequencies over tasks, so ÷ by number of tasks
                normed = freq / score.size
                eval_stats[model][mode]["calibration_overall"] = float(
                    ((normed - midpoints) ** 2).sum()
                )
            if config["sharpness"]:
                sharpness = numpy.array(tracked_stats[model][mode]["sharpness"])
                mean, stderr = sharpness.mean(), sharpness.std() / sharpness.size ** 0.5
                eval_stats[model][mode]["sharpness"]["mean"] = float(mean)
                eval_stats[model][mode]["sharpness"]["stderr"] = float(stderr)

    return eval_stats
