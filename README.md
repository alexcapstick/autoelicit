<p align="center">
<img src="https://github.com/alexcapstick/autoelicit/raw/main/assets/autoelicit_logo.png" alt="AutoElicit logo" width="400"/>
</p>

# AutoElicit: Using Large Language Models for Expert Prior Elicitation in Predictive Modelling

This python package, `autoelicit`, can be used to elicit priors on linear models from large language models.

This code is based on the method presented in the paper <a href="https://arxiv.org/abs/2411.17284" target="_blank">AutoElicit: Using Large Language Models for Expert Prior Elicitation in Predictive Modelling</a>


The code to reproduce the experiments in the paper can be found in the repo <a href="https://github.com/alexcapstick/llm-elicited-priors" target="_blank">llm-elicited-priors</a>, whilst this repo contains the code to use the method in practice.

Our method uses large language models (LLMs) to provide the mean and standard deviation of many Gaussian priors which can be placed on the parameters of a linear model. By sampling the LLM multiple times, we form a mixture of these Gaussians, which can be used to construct linear predictive models or analysed to understand the risk factors for the target variable.

From our paper:

Given an LLM $M$ and task $T$, we obtain a single Gaussian prior for each feature, for each task description $I_k$ by asking the LLM to provide its best guess of the mean and standard deviation: $(\mu_k, \sigma_k) \sim \Pr_{M, T} (\mu, \sigma | I_k)$.
Here, we use $K=100$ task descriptions, produced by asking an LLM to rephrase one human-written system and user role $10$ times and taking their product.

Taking a mixture over the task descriptions, we construct a prior $\Pr_{M,T}(\theta)$ over linear model parameters $\theta$:
<p align="center">
$$\Pr_{M, T}(\theta) = \sum_{k=1}^K \pi_k \mathcal{N} ( \theta | \mu_k, {\sigma_k}^2 ) ~ ~ \text{where} ~ (\mu_k, \sigma_k) \sim  \Pr_{M, T} (\mu, \sigma | I_k) ~ \text{and} ~ \pi_k \sim \text{Dir} (1)$$
</p>


## Installation

To install the package, run the following command:

```bash
pip install autoelicit
```

## Usage

We provide an example for how to use this package in the [example notebook](./examples/elicit_priors_for_a_toy_dataset.ipynb) and a shorter example in the [example script](./examples/elicit_priors_for_a_toy_dataset.py), but the basic usage is as follows:


```python
# import the elicitation function
from autoelicit.gpt import (
    get_llm_elicitation_for_dataset
)

# elicit priors for a dataset
get_llm_elicitation_for_dataset(
    # the language model client
    # requires a .get_result(messages)
    client=llm_client,
    # the task descriptions
    # to iterate over
    system_roles=system_roles,
    user_roles=user_roles,
    # the dataset feature names
    feature_names=data.feature_names,
)
```

A complete example with the loading of a dataset and the prompts from some directory might look like:

```python
# import the necessary functions and classes
from autoelicit.utils import load_prompts
from autoelicit.gpt import GPTOutputs, get_llm_elicitation_for_dataset
# a toy dataset to demonstrate the method
from autoelicit.datasets import load_breast_cancer

# wrapper for language models
# see the notebook linked above for more details
CLIENT_CLASS = GPTOutputs
CLIENT_KWARGS = dict(
    temperature=0.1,
    model_id="gpt-3.5-turbo-0125",
    result_args=dict(
        response_format={"type": "json_object"},
    ),
)

# load the dataset which contains information
# about the feature names, target names, and 
# the dataset itself
dataset = load_breast_cancer()

# load the prompts for the system and user roles
system_roles = load_prompts("prompts/elicitation/system_roles_breast_cancer.txt")
user_roles = load_prompts("prompts/elicitation/user_roles_breast_cancer.txt")

# create the llm client
client = CLIENT_CLASS(**CLIENT_KWARGS)

#### elicit the priors for the dataset ####
expert_priors = get_llm_elicitation_for_dataset(
    # the language model client
    client=client,
    # the prompts
    system_roles=system_roles,
    user_roles=user_roles,
    # the dataset contains the feature names as an attribute
    feature_names=dataset.feature_names.tolist(),
    # the dataset contains the target names as an attribute
    target_map={k: v for v, k in enumerate(dataset.target_names)},
    # print the prompts before passing them to the language model
    verbose=True,
)
```


The example in the notebook would then provide a prior distribution over the parameters such as:


<p align="center">
<img src="https://github.com/alexcapstick/autoelicit/raw/main//examples/figures/breast_cancer_prior_elicitation.png" alt="AutoElicit logo" style="max-width:750px" width="100%" />
</p>


These prior parameter values can be studied as risk factors for the target variable, and can be used to inform the design of a predictive model. <a href="https://github.com/alexcapstick/llm-elicited-priors" target="_blank">In our experiments</a> we use `pymc` with these priors, but other methods could be used.


For example, the prior above suggests that the feature `Mean Area` is a strong positive risk factor for the target variable `Malignant`, whilst the feature `Mean Smoothness` is a negative risk factor. 

## Documentation

The documentation for all of the functions in the package is in development.


## Citation

If you find this work useful, please consider citing our preprint:

```
@article{capstick2024autoelicit,
  title={Auto{E}licit: {U}sing Large Language Models for Expert Prior Elicitation in Predictive Modelling},
  author={Capstick, Alexander and Krishnan, Rahul G and Barnaghi, Payam},
  journal={International Conference on Machine Learning (ICML)},
  year={2025},
  url={https://doi.org/10.48550/arXiv.2411.17284}
}
```
