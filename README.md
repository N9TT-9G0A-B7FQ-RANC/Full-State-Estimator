# Learning Full State Estimator from Physics Principles via the Moving Horizon Estimation Framework

## Install Requirements

To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt

## Generate Dataset
To generate the required dataset for training the model, execute the following commands:

```bash
python ./system/simulate_duffing_oscillator.py
python ./system/simulate_vanderpol.py

## Launch Training
To train the model, use the provided run.sh script. This script contains the necessary commands to initiate the training process.

```bash
./run_train.sh

## Evaluate Obtained Models
After training, you can evaluate the performance of the obtained models using the run_evaluate.sh script.

```bash
./run_evaluate.sh