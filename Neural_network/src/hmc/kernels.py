import numpy as np
import tensorflow_probability as tfp
tfd = tfp.distributions

def get_hmc_kernel(target_log_prob_fn, step_size, num_leapfrog_steps):
    hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob_fn,
        step_size=step_size,
        num_leapfrog_steps=num_leapfrog_steps)
    return hmc_kernel

def get_nuts_kernel(target_log_prob_fn, step_size):
    nuts_kernel = tfp.mcmc.NoUTurnSampler(
        target_log_prob_fn=target_log_prob_fn,
        step_size=step_size)
    return nuts_kernel

def init_step_adapter(step_adapter, inner_kernel, num_adaptation_steps):
    if type(inner_kernel)==tfp.mcmc.HamiltonianMonteCarlo:
        adaptive_step_kernel = step_adapter(
            inner_kernel=inner_kernel,
            num_adaptation_steps=num_adaptation_steps)

    elif type(inner_kernel)==tfp.mcmc.NoUTurnSampler:
        adaptive_step_kernel = step_adapter(
            inner_kernel=inner_kernel,
            num_adaptation_steps=num_adaptation_steps,
            step_size_setter_fn=lambda pkr, new_step_size: pkr._replace(step_size=new_step_size),
            step_size_getter_fn=lambda pkr: pkr.step_size)#,
            # log_accept_prob_getter_fn=lambda pkr: pkr.log_accept_ratio)

    return adaptive_step_kernel