import textwrap
import numpy as np

from autoelicit import datasets
from autoelicit import get_llm_elicitation_for_dataset

data = datasets.load_breast_cancer()

system_roles = [
"""
You are a simulator of a logistic regression predictive model 
for predicting breast cancer diagnosis from tumour characteristics.
Here the inputs are tumour characteristics and the output is 
the probability of breast cancer diagnosis from tumour characteristics. 
Specifically, the targets are ['benign', 'malignant'] with mapping 
['benign' = 0, 'malignant' = 1].
With your best guess, you can provide the probabilities of a malignant 
breast cancer diagnosis for the given tumour characteristics.
"""
]

user_roles = [
"""
I am a data scientist working with a dataset aimed at predicting
breast cancer diagnoses based on tumor characteristics. I would like
to utilize your model to forecast the diagnosis for my samples. My
dataset includes the following features: {feature_names}. All feature
values have been standardized using the z-score method. By considering
how each feature might relate to a diagnosis of ['benign',
'malignant'] and whether each one shows a positive or negative
correlation with the outcomes of ['benign' = 0, 'malignant' = 1], I
would appreciate it if you could estimate the mean and standard
deviation for a normal distribution prior for each feature that could
be applied in a logistic regression model for predicting breast cancer
diagnosis from tumor characteristics. Please return your response as a
JSON object with feature names as keys and a nested dictionary
containing mean and standard deviation as values. A positive mean
suggests a positive correlation with the outcome, a negative mean
indicates a negative correlation, and a small standard deviation
reflects your confidence in your estimate. Please provide only a JSON
response, without any additional text.
""",

"""
I am a data scientist
with a dataset tasked with predicting breast cancer diagnoses from
tumor characteristics. I would like to apply your model to forecast
the diagnosis of my samples. The dataset I have includes the following
features: {feature_names}. All feature values have been standardized
using the z-score. By evaluating how each feature may relate to a
diagnosis of ['benign', 'malignant'] and whether each feature
correlates positively or negatively with the diagnosis outcomes
['benign' = 0, 'malignant' = 1], I request that you estimate the mean
and standard deviation for a normal distribution prior for each
feature to be used in a logistic regression model for predicting
breast cancer diagnosis based on tumor characteristics. Kindly respond
with a JSON object where the feature names are the keys, and the
corresponding mean and standard deviation are nested as values. A
positive mean indicates a positive correlation with the outcome, while
a negative mean signals a negative correlation; furthermore, a small
standard deviation indicates a high level of confidence in your
guesses. Please limit your response to JSON only, without any other
information.
""",

"""
As a data scientist, my current task involves using
a dataset to predict breast cancer diagnoses from tumor
characteristics. I am seeking to employ your model to determine the
diagnosis of my samples. The dataset consists of the following
features: {feature_names}. All values of the features are standardized
using the z-score method. Considering how each feature might correlate
with a diagnosis of ['benign', 'malignant'] and whether they are
positively or negatively associated with the outcomes of ['benign' =
0, 'malignant' = 1], I would like you to provide estimates for the
mean and standard deviation for a normal distribution prior for each
feature suitable for a logistic regression model dedicated to
predicting breast cancer diagnosis from tumor characteristics. Please
reply with a JSON object where the keys are the feature names, and the
values are nested dictionaries of mean and standard deviation. A
positive mean suggests a positive correlation with the outcome, while
a negative mean implies a negative correlation, and a smaller standard
deviation reflects greater confidence in your estimation. Please
respond with only JSON, avoiding any additional text.
"""
]

from autoelicit.gpt import (
    GPTOutputs, LlamaOutputs, QwenOutputs, DeepSeekOutputs
)

client = GPTOutputs(
    model_id="gpt-3.5-turbo",
    # setting the temperature of the model
    temperature=0.1,
    # when performing prior elicitation with LLMs, we 
    # must specify that we want a json object returned.
    result_args=dict(
        response_format={"type": "json_object"},
    ),
    rng=np.random.default_rng(42),
)

priors = get_llm_elicitation_for_dataset(
    client=client,
    # we would usually use multiple system and user roles
    # so that we can build a mixture of the priors elicited.
    system_roles=system_roles,
    user_roles=user_roles,
    # this will replace the {feature_names}
    # in the prompts and also allow us to correctly
    # convert the JSON response into a numpy array 
    # of means and stds
    feature_names=data.feature_names.tolist(),
    # prints progress, each prior elicited, and the prompts
    # used to elicit the priors
    verbose=True
)

print("The shape of the prior is: ", priors.shape)

print("The priors are: \n", priors)