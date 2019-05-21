# CIS700-BERT-RACE

The work presented below was a joint effort of Peyman Norouzi, Samuel Oshay, Leonardo Murri, and Dewang Sultania.

## Setup for Logistic Regresion (Non-Deep Baseline)
1. Download and unzip https://bit.ly/2GE1Frw into Logistic_Regression/Glove/
2. Unzip Data.zip in Logistic_Regression/
3. From Logistic_Regression, run python3 run.py

## Setup for CNN (Deep Baseline #1)

### To evaluate the model:

1. Download and unzip the data from https://bit.ly/2VvxDyp into CNN/
2. Download and unzip https://bit.ly/2GE1Frw into CNN/ .The files weight_matrix.pickle and word_to_idx.pickle should be directly under the CNN folder.
3. Download https://bit.ly/2GS5idL and put it in CNN/
4. Run cnn_eval.py

### To train the model:

1. Download and unzip the data from https://bit.ly/2VvxDyp into CNN/
2. Download and unzip https://bit.ly/2GE1Frw into CNN/ .The files weight_matrix.pickle and word_to_idx.pickle should be directly under the CNN folder.
3. Run cnn.py

## Setup for Feed Forward Net (Deep Baseline #2)

### To train and evaluate the model:
1. Download and unzip data from https://bit.ly/2IMpAY5
2. Run feed_forward.py

## Setup for GRU (Deep Baseline #3)

### To evauluate the model:
1. Download and unzip data from https://bit.ly/2VrGPE8 into GRU/
2. Download the trained model https://bit.ly/2DyVB1t into GRU/
3. Run gru_eval.py

### To train the model:
1. Download and unzip data from https://bit.ly/2VrGPE8 into GRU/
2. Run gru.py

## Setup for BERT (Advanced Deep #1)

### To evaluate the model:

1. Download and unzip the trained model from https://bit.ly/2XDOe0k to a folder called large_models/
2. run python3 evaluate_bert.py

### To train the model

1. Run python3 race_bert.py it will run your model and save it in a folder called large_models/
2. Run evaluate_bert.py post that to run it in test set.

## Setup for DCMN (Advanced Deep #2 / Novel Extra Credit)

### To train and evaluate the model:

1. Download train, dev, and test data from https://bit.ly/2PF5v6Q.
2. Ensure pytorch-pretrained-bert is installed.
3. Run python DCMN.py where a GPU is installed and available.

## Data Hypotheses

The folder Hypo contains all the hypothesis we tested, details regarding those are in the slides (Attention class.pdf), technical report and the blog post.

