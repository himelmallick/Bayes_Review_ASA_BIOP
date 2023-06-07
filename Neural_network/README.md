BNNs for Tox21
==============================
### Tatiana Matejovicova, September 2020 ###

Using BNNs to predict toxicity on Tox21 data. Upon request, we can provide the entire repository to run the models. Currently, the files are uploded in src directory for the priliminary codes.

Project Organization
------------

    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── interim        <- Intermediate data that has been transformed
    │   ├── processed      <- The final, canonical data sets for modeling
    │   └── raw            <- The original, immutable data dump
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── docs               <- Documents that contain information about this project
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    └── src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        │
        ├── data           <- Scripts to process data
        │
        ├── hmc            <- Code to define hmc and run sampling
        │   └── run.py
        │
        ├── nn             <- Code to build and run neural networks
        │   └── run.py
        │
        └── utils          <- Helper functions


--------
Note that the models used to obtain BNN results in the presentation ```docs/BNNs_for_Toxicity_Modelling_Tatiana_Matejovicova.pptx``` (slide 12) are saved in ```/projects/qbio/ml_ai/kxkr044/bnns_for_tox21_results/models``` and can be reproduced.

## 1. Installation ##
* The code was written in Python 3.6 with the use of TensorFlow 2.2 and Tensorflow Probability 0.10.0
* Therefore some functionality might not work as expected with previous versions 
* Create a virtual environment for example by running:
```
conda create -n myenv python=3.6
conda activate myenv
```
* In the terminal, move to the project directory and run:
```
pip install -r requirements.txt
pip install --editable .
```

## 3. Hamiltonian Monte Carlo ##
* To define HMC and run sampling run the following:
```
python src/hmc/run.py --config CONFIG_FILE
```
* Specify ```CONFIG_FILE```, a path to a config file in the json format, which specifies the details about the model
* For example:
```
python src/hmc/run.py --config models/hmc/layers2/run1/config.json
```
* Inspect ```src/hmc/run.py``` to find out what the different configuration options are

## 3. Neural Network ##
* To build and run neural networks run the following:
```
python src/nn/run.py --config CONFIG_FILE
```
* Specify ```CONFIG_FILE```, a path to a config file in the json format, which specifies the details about the model
* For example:
```
python src/hmc/run.py --config models/nn/layers2/run1/config.json
```
* Inspect ```src/nn/run.py``` to find out what the different configuration options are

## 4. Data ##
* The canonical datasets with RDKit Phys-Chem features, for the Mitochondrial Membrane Potential Assay are as follows:
   * Train features: ```data/tox21/processed/chem/sr/mmp/scaled_train_chem.csv```
   * Train labels: ```data/tox21/processed/chem/sr/mmp/train_activity.csv```
   * Validation features: ```data/tox21/processed/chem/sr/mmp/scaled_val_chem.csv```
   * Validation labels: ```data/tox21/processed/chem/sr/mmp/val_activity.csv```
   * Test features: ```data/tox21/processed/chem/sr/mmp/scaled_test_orig_chem.csv```
   * Test labels: ```data/tox21/processed/chem/sr/mmp/test_orig_activity.csv```
* The test is equivalent to that from the Tox 21 challenge
* The mapping from compound ids to smiles after data cleaning is ```data/tox21/interim/2_full/3_merged/smiles.csv```

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
