# Computational Intelligence Lab 2024 <br /> Twitter Sentiment Classification

Lorenzo Benedetti, Tommaso Di Mario, Andrea Sgobbi,  Gabriel Tavernini

## Problem Statement
Given a set of **Tweets**, predict *positive*/*negative* sentiment. The labels are assigned based on the presence of positive `:)` or negative `:(` emojis, hence there is non-negligible label noise (sarcasm) and the possibility of different labels for the same Tweet.

## Environment Setup
The environment is managed through `virtualenv` and uses `python 3.8`:

```
$ python -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

The project directory is structured as follows:

```
cil/
├── src/                               # All source code
│   ├── checkpoints/                   # Training checkpoints
│   │   └── ...
│   ├── ensemble/                      # Model weights used in LLM ensemble
│   │   ├── finetuned_model/
│   │   │   └── model.safetensors
│   │   └── adapter_model/
│   │       ├── adapter_model.safetensors
│   │       └── adapter_config.json
│   ├── pretrained_embeddings/         # Embeddings used in baselines
│   │   └── ...
│   ├── config.yaml                    # Global config used by all models
│   ├── ensemble.py                    # Compute outputs for models in LLM ensemble
│   ├── ensemble.ipynb                 # Train mixing model
│   └── ...                            # Scripts to train and test specific baselines
├── twitter-datasets/                  # Full/Subsampled Twitter datasets
│   └── ...
├── requirements.txt                   # pip requirements file
└── README.md                          # Project documentation
```

## Running Models
Each individual model can be trained and tested using its corresponding script, such as `svm.py` or `llm.py`. This will output a `<config.general.run_id>.csv` file in the format of the Kaggle submission.

Each model has a separate section in `config.yaml` which needs to be configured before execution.

## Running Ensembles
For ensembles, the practice is slightly more involved as we assume the individual models to be already trained. Each model in the ensemble needs to be explicitly added to `config.ensemble.models`. The `name` field should correspond to a folder in the `src/ensemble` directory. Note that the files we expect differ between models being fully finetuned and models being trained with *LoRa* (refer to directory structure)

The script `ensemble.py` will run inference on the ensamble models and save the outputs as 3 `.npy` files. These files are then used within ```ensemble.ipynb``` to train the final mixing 'layer' of the ensemble model.