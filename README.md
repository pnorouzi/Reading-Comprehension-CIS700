# CIS700-SQuAD-RACE

## Setup for Logistic Regresion
1. Download and unzip https://bit.ly/2GE1Frw into Logistic_Regression/Glove/
2. Unzip Data.zip in Logistic_Regression/
3. From Logistic_Regression, run python3 run.py

## Setup for CNN

### To evaluate the model:

1. Download and unzip the data from https://bit.ly/2VvxDyp into CNN/
2. Download and unzip https://bit.ly/2GE1Frw into CNN/ .The files weight_matrix.pickle and word_to_idx.pickle should be directly under the CNN folder.
3. Download https://bit.ly/2GS5idL and put it in CNN/
4. Run cnn_eval.py

### To train the model:

1. Download and unzip the data from https://bit.ly/2VvxDyp into CNN/
2. Download and unzip https://bit.ly/2GE1Frw into CNN/ .The files weight_matrix.pickle and word_to_idx.pickle should be directly under the CNN folder.
3. Run cnn.py

## Setup for BERT

### To evaluate the model:

1. Download and unzip the trained model from https://bit.ly/2XDOe0k to a folder called large_models/
2. run python3 evaluate_bert.py

### To train the model

1. Run python3 race_bert.py it will run your model and save it in a folder called large_models/
2. Run evaluate_bert.py post that to run it in test set.

## Setup for DCMN

### To train and evaluate the model:

1. Download train, dev, and test data from https://bit.ly/2PF5v6Q.
2. Ensure pytorch-pretrained-bert is installed.
3. Run python DCMN.py where a GPU is installed and available.

## Data Hypotheses

The folder Hypo contains all the hypothesis we tested, details regarding those are in the slides (Attention class.pdf), technical report and the blog post.


