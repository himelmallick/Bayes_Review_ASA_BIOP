import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

@tf.function
def sample_chains(num_results, init_state, kernel, num_burnin_steps, trace_fn):
    states, kernel_results = tfp.mcmc.sample_chain(
        num_results=num_results,
        current_state=init_state,
        kernel=kernel,
        num_burnin_steps=num_burnin_steps,
        trace_fn=trace_fn)
    return states, kernel_results

def get_trace_everything_fn():
    def trace_everything(_, previous_kernel_results):
        return previous_kernel_results
    return trace_everything

def get_trace_nothing_fn():
    def trace_nothing(*args):
        return {}
    return trace_nothing

def get_trace_subset_fn(kernel):
    if type(kernel)==tfp.mcmc.SimpleStepSizeAdaptation or type(kernel)==tfp.mcmc.DualAveragingStepSizeAdaptation:
        inner_kernel = kernel.inner_kernel
        if type(inner_kernel)==tfp.mcmc.HamiltonianMonteCarlo:
            def trace_subset(_, pkr):
                return {
                    "target_log_prob": pkr.inner_results.accepted_results.target_log_prob,
                    "is_accepted": pkr.inner_results.is_accepted,
                    "log_accept_ratio": pkr.inner_results.log_accept_ratio,
                    "step_size": pkr.inner_results.accepted_results.step_size}
            return trace_subset

        elif type(inner_kernel)==tfp.mcmc.NoUTurnSampler:
            def trace_subset(_, pkr):
                return {
                    "target_log_prob": pkr.inner_results.target_log_prob,
                    "is_accepted": pkr.inner_results.is_accepted,
                    "log_accept_ratio": pkr.inner_results.log_accept_ratio,
                    "leapfrogs_taken": pkr.inner_results.leapfrogs_taken,
                    "step_size": pkr.inner_results.step_size}
            return trace_subset

    elif type(kernel)==tfp.mcmc.HamiltonianMonteCarlo:
        def trace_subset(_, pkr):
            return {
                "target_log_prob": pkr.accepted_results.target_log_prob,
                "is_accepted": pkr.is_accepted,
                "log_accept_ratio": pkr.log_accept_ratio}
        return trace_subset

    elif type(kernel)==tfp.mcmc.NoUTurnSampler:
        def trace_subset(_, pkr):
            return {
                "target_log_prob": pkr.target_log_prob,
                "is_accepted": pkr.is_accepted,
                "log_accept_ratio": prk.log_accept_ratio,
                "leapfrogs_taken": pkr.leapfrogs_taken}
        return trace_subset