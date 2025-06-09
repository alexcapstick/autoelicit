from setuptools import setup

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='autoelicit',
    version='0.1.14',    
    description='A python package for eliciting prior knowledge from experts using large language models.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/alexcapstick/autoelicit',
    author='Alexander Capstick',
    author_email='alexander.capstick19@imperial.ac.uk',
    license='MIT',
    packages=['autoelicit'],
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "scikit-learn",
        "tqdm",
        "pyarrow",
        "ucimlrepo",
        "openai>=1.51.2"        
    ],
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)