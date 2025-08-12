# Chagas_Challenge_Team_UdeC

## Execution Instructions

This repository follows the **same structure and command-line interface** as the official [PhysioNet Challenge starter code](https://github.com/physionetchallenges/python-example-2024). The workflow is compatible with the automated evaluation system, including Docker-based execution.

### Train the model

Use the following command to train your model:

```bash
python train_model.py -d training_data -m model
````
Run the model:
````
python run_model.py -d holdout_data -m model -o holdout_outputs
````
Evaluate the model
````
python evaluate_model.py -d holdout_data -o holdout_outputs -s scores.csv
