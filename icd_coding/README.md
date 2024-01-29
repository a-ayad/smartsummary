# Multi-Label Text Classification for ICD coding
## Directory Structure
```sh
.
├── README.md
├── attention
│   ├── __init__.py
│   └── attention_layer.py
├── base
│   ├── __init__.py
│   └── base_model.py
├── data_loader
│   ├── __init__.py
│   ├── mimic_data_loader.py
│   └── utils.py
├── mimicdata
│   ├── D_ICD_DIAGNOSES.csv
│   ├── D_ICD_PROCEDURES.csv
│   ├── ICD9_descriptions
│   ├── TOP_30_CODES.csv
│   ├── TOP_50_CODES.csv
│   ├── dev_30.csv
│   ├── dev_50.csv
│   ├── test_30.csv
│   ├── test_50.csv
│   ├── train_30.csv
│   ├── train_50.csv
│   └── vocab.csv
├── models
│   ├── __init__.py
│   ├── cnn.py
│   ├── lstm.py
│   ├── multi_cnn.py
│   ├── rcnn.py
│   └── tcn.py
├── run_experiment.py
└── training.sh
```
We can see that the main breakdown of the codebase is between attention, base, data_loader, mimicdata and models.

### Attention
The package contains our custom attention layers.

### Base
We create a base model for all the models, which has a few functions we are often to use:

- `build_model`: constructing the neural networks
- `save`: saving the weights of models in a specified filepath
- `load`: loading the weights from a filepath
- `predict`: the same predict function as Tensorflow API
- `evaluate`: the same evaluate function as Tensorflow API 

### Data_Loader
The `MimicDataLoader` class in the `mimic_data_loader.py` is able to load the training/validation/test data and preprocess the data to get it ready to our models. 

### mimicdata
This folder contains all the data we use in our experiment.

### Models
Each model has a file for it, you can edit the architecture or add specified command-line flags for each model.

## Experiment
We use a `ReduceLROnPlateau` callback to automatically reduce the learning rate by 10% if there is no improvement in every 4 epochs. 
We also use an `EarlyStopping` mechanism, in which the training is stopped if there is no improvement of the micro-averaged F1 score on the validation set in 6 continuous epochs.
And the weights of the model would be saved at the epoch which has the best mirco-averaged F1 score on the validation set. 
## Training
The `run_experiment.py` is a script that handles some command-line parameters and also picks up from the model that is specified.
For example, the `CNN` model has a couple of command-line flags: `--cnn_filters` and `--cnn_kernel_size`. If you want to use the attention layer we build in the `attention` package, only need to add the `--use_attention` flag.

Here is an example of `CNN with attention`
```sh
python \
run_experiment.py \
--model_class=CNN \
--learning_rate=0.001 \
--embedding_dim=100 \
--cnn_filters=500 \
--cnn_kernel_size=4 \
--batch_size=8 \
--dropout=0.2 \
--epochs=50 \
--max_length=2500 \
--use_attention
```

After the training is finished, the checkpoint in the best micro-averaged F1 score on the validation set would be saved, and the F1 score evaluated by the saved model in test data set would be shown.

## Evaluate
If you only want to check your saved model, you can just use the same command-lin flags as your saved model. Then you only need to add one more flag `test_model` for your saved checkpoint.

For example, we can run
```sh
python \
run_experiment.py \
--model_class=CNN \
--learning_rate=0.001 \
--embedding_dim=100 \
--cnn_filters=500 \
--cnn_kernel_size=4 \
--batch_size=8 \
--dropout=0.2 \
--epochs=50 \
--max_length=2500 \
--use_attention
--test_model=ckpt_filepath
```

## Results
| Models | Attention | val_f1_macro | val_f1_micro | test_f1_macro | test_f1_micro |
| :----: | :-------: | :---------:  | :----------: | :----------:  | :-----------: |
| CNN | | 0.56 | 0.65 | 0.55 | 0.64 |
| CNN | ✅ | 0.56 | 0.65 | 0.55 | 0.65 |
| LSTM | | 0.49 | 0.60 | 0.49 | 0.60 |
| LSTM | ✅ | 0.55 | 0.66 | 0.54 | 0.65 |
| **MultiCNN** | | **0.61** | **0.67** | **0.59** | 0.65 | 
| MultiCNN | ✅ | 0.57 | 0.65 | 0.57 | 0.65 |
| RCNN | | 0.55 | 0.64 | 0.54 | 0.64 |
| RCNN | ✅ | 0.56 | 0.65 | 0.56 | 0.65 |
| TCN |  | 0.43 | 0.56 | 0.43 | 0.56 |
| TCN | ✅ | 0.54 | 0.63 | 0.54 | 0.63 |

