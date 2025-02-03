"""
autoelicit.

A python package for eliciting prior knowledge from experts.
"""

__version__ = "0.1.2"
__author__ = 'Alexander Capstick'


from .gpt import (
    get_llm_elicitation_for_dataset, 
    get_llm_elicitation, 
    rephrase_task_description
)