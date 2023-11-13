import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import src.hmc.eval as eval
import src.hmc.kernels as kernels
import src.hmc.priors as priors
import src.hmc.sample as sample
from src.hmc.target import Target
from src.hmc.tied_target import TiedTarget
from src.hmc.tied_target_global import TiedTargetGlobal
import src.utils.helpers as helpers

tfd = tfp.distributions

def main(config_arg):
    config_file = Path(config_arg).absolute()
    if not config_file.is_file():
        raise Exception("Config file doesn't exist.")
    config = helpers.read_config(config_file)
    run_hmc(config)

def run_hmc(config):
    # Reset tensorflow session
    # tf.keras.backend.clear_session()

    output_dir_path = Path(config["output"]["dir"]) # Where all the output is saved

    # 0. Redirect output
    if config["output"]["file"] is not None:
        output_file_path = output_dir_path.joinpath(config["output"]["file"])
        sys.stdout = open(output_file_path,'wt')
    # Get pid
    print("Process id is", os.getpid())

    # 1. Load data
    X_train = helpers.read_df(config["data"]["train"]["features"]).values.astype(np.float32)
    y_train = helpers.read_df(config["data"]["train"]["labels"]).values.astype(np.int).squeeze()
    X_val = helpers.read_df(config["data"]["val"]["features"]).values.astype(np.float32)
    y_val = helpers.read_df(config["data"]["val"]["labels"]).values.astype(np.int).squeeze()
    X_test = helpers.read_df(config["data"]["test"]["features"]).values.astype(np.float32)
    y_test = helpers.read_df(config["data"]["test"]["labels"]).values.astype(np.int).squeeze()

    # 2. Obtain target log probability function
    layers = config["model"]["target"]["layers"]

    # Select prior
    prior_distribution = config["model"]["target"]["prior"]["distribution"]
    scale = config["model"]["target"]["prior"]["scale"]
    if prior_distribution=="normal":
        prior = priors.get_normal_prior(scale)
    elif prior_distribution=="laplace":
        prior = priors.get_laplace_prior(scale)
    elif prior_distribution=="horseshoe":
        prior = priors.get_horseshoe_prior(scale)

    # Generate target
    if config["model"]["target"]["tied"]:
        if config["model"]["target"]["fix_global_shrinkage"]:
            target = TiedTargetGlobal(X_train, y_train, layers, prior, config["model"]["target"]["global_shrinkage"])
        else:
            target = TiedTarget(X_train, y_train, layers, prior)
    else:
        target = Target(X_train, y_train, layers, prior)
    target_log_prob_fn = target.get_target_log_prob_fn()

    # 3. Step size
    if config["model"]["kernel"]["step_size_per_chain"]:
        step_size = tf.ones((config["model"]["sample"]["num_chains"], 1))*config["model"]["kernel"]["step_size"]
    else:
        step_size = config["model"]["kernel"]["step_size"]

    # 4. Load kernel
    kernel_name = config["model"]["kernel"]["kernel_name"]
    if kernel_name=="hmc":
        kernel = kernels.get_hmc_kernel(target_log_prob_fn, step_size=step_size, num_leapfrog_steps=config["model"]["kernel"]["num_leapfrog_steps"])
    elif kernel_name=="nuts":
        kernel = kernels.get_nuts_kernel(target_log_prob_fn, step_size=step_size)

    # 5. Step size adaptation
    step_adapter_name = config["model"]["kernel"]["step_adapter_name"]
    if step_adapter_name=="simple":
        step_adapter = tfp.mcmc.SimpleStepSizeAdaptation
    elif step_adapter_name=="dual_averaging":
        step_adapter = tfp.mcmc.DualAveragingStepSizeAdaptation
    elif step_adapter_name is None:
        step_adapter = None
    if step_adapter is not None:
        kernel = kernels.init_step_adapter(step_adapter, kernel, num_adaptation_steps=config["model"]["kernel"]["num_adaptation_steps"])

    # 6. Trace function
    trace = config["model"]["sample"]["trace"]
    if trace=="everything":
        trace_fn = sample.get_trace_everything_fn()
    elif trace=="subset":
        trace_fn = sample.get_trace_subset_fn(kernel)
    elif trace==None:
        trace_fn = sample.get_trace_nothing_fn()

    # 7. Init state
    if config["model"]["target"]["tied"]:
        if config["model"]["target"]["tied_init_state"]=="fixed":
            init_state = target.get_fixed_shrinkage_init_state(num_chains=config["model"]["sample"]["num_chains"])
        elif config["model"]["target"]["tied_init_state"]=="sampled":
            init_state = target.get_sampled_shrinkage_init_state(num_chains=config["model"]["sample"]["num_chains"])
        elif config["model"]["target"]["tied_init_state"]==None:
            init_state = target.get_init_state(num_chains=config["model"]["sample"]["num_chains"])
    else:
        init_state = target.get_init_state(num_chains=config["model"]["sample"]["num_chains"])
    
    # 8. Sample chains
    if config["model"]["eval"]["load_states_file"] is None:
        start_time = time.time()
        states, kernel_results = sample.sample_chains(
            num_results=config["model"]["sample"]["num_results"],
            init_state=init_state,
            kernel=kernel,
            num_burnin_steps=config["model"]["sample"]["num_burnin_steps"],
            trace_fn=trace_fn)
        time_elapsed = time.time() - start_time
        print("Sampling took: {0:.1f} seconds.\n".format(time_elapsed))
    else:
        states_file = Path(config["model"]["eval"]["load_states_file"]).absolute()
        states = helpers.read_pickle(states_file)

    # 9. Save states and kernel results
    if config["model"]["eval"]["save_states"]:
        states_file = output_dir_path.joinpath("states.pickle").absolute()
        helpers.write_pickle(states, states_file)

    if config["model"]["eval"]["save_kernel_results"] and config["model"]["eval"]["load_states_file"] is None:
        kernel_results_file = output_dir_path.joinpath("kernel_results.pickle").absolute()
        helpers.write_pickle(kernel_results, kernel_results_file)

    # 10. Evaluate model
    print("Evaluation:")
    skip_burnin_steps = config["model"]["eval"]["skip_burnin_steps"]
    states = states[skip_burnin_steps:, :, :]
    post_samples_train, post_means_train = eval.get_posterior(X_train, states, target)
    eval.eval_hmc(y_train, post_means_train, text="Train data")

    post_samples_val, post_means_val = eval.get_posterior(X_val, states, target)
    eval.eval_hmc(y_val, post_means_val, text="Val data")

    post_samples_test, post_means_test = eval.get_posterior(X_test, states, target)
    eval.eval_hmc(y_test, post_means_test, text="Test data")
    
    # 11. Save predictions
    if config["model"]["eval"]["save_predictions"]:
        helpers.write_pickle(post_samples_train, output_dir_path.joinpath("post_samples_train.pickle").absolute())
        helpers.write_pickle(post_means_train, output_dir_path.joinpath("post_means_train.pickle").absolute())
        helpers.write_pickle(post_samples_val, output_dir_path.joinpath("post_samples_val.pickle").absolute())
        helpers.write_pickle(post_means_val, output_dir_path.joinpath("post_means_val.pickle").absolute())
        helpers.write_pickle(post_samples_test, output_dir_path.joinpath("post_samples_test.pickle").absolute())
        helpers.write_pickle(post_means_test, output_dir_path.joinpath("post_means_test.pickle").absolute())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Specify config file')
    args = parser.parse_args()
    main(args.config)
