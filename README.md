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
│   ├── pretrained_embeddings/
│   │   ├── glove.25d.txt
│   │   ├── glove.50d.txt
│   │   ├── glove.100d.txt
│   │   └── glove.200d.txt
│   ├── config.yaml                    # Global config used by all models
│   ├── baselines.py                   # Train and test Logistic/RandomForest/SVM classifiers
│   ├── neural_baselines.py            # Train and test FFNN/CNN/LSTM networks
│   ├── gzip_knn.py                    # Test the GZIP K-Nearest-Neighbor classifier
│   ├── llm.py                         # Train and test LLM classifiers
│   ├── ensemble.py                    # Compute outputs for models in LLM ensemble
│   ├── ensemble.ipynb                 # Train mixing model
│   └── ...                            # Model-specific scripts
├── twitter-datasets/
│   ├── test_data.txt
│   ├── train_neg_full.txt
│   ├── train_neg.txt
│   ├── train_pos_full.txt
│   └── train_pos.txt
├── requirements.txt                   # pip requirements file
└── README.md                          # Project documentation
```

## Running Models
Each individual model can be trained and tested using the script specified in the directory structure. This will output a `<config.general.run_id>.csv` file in the format of the Kaggle submission.

Each model has a separate section in `config.yaml` to configure hyperparameters. The specific model used by `baselines.py` and `neural_baselines.py` can then be selected like so:
* `config.baselines.model` should be set to *'logit'* for Logistic Regression, *'rf'* for Random Forest or *'svm'* for the Support Vector Machine.
* `config.baselines.data` should be set to *'count'* to use word count embeddings, *'tfidf'* for TF-IDF or *'glove'* for Glove embeddings.
* `config.neural_baselines.model` should be set to *'ffnn'* for the Feed-Forward NN, *'cnn'* for the CNN or *'lstm'* for the LSTM.

## Running Ensembles
For ensembles, the practice is slightly more involved as we assume the individual models to be already trained. Each model in the ensemble needs to be explicitly added to `config.ensemble.models`. The `name` field should correspond to a folder in the `src/ensemble` directory. Note that the files we expect differ between models being fully finetuned and models being trained with *LoRa* (refer to directory structure)

The script `ensemble.py` will run inference on the ensamble models and save the outputs as 3 `.npy` files. These files are then used within ```ensemble.ipynb``` to train the final mixing 'layer' of the ensemble model.