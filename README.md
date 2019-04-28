# CIS700-SQuAD-RACE

## Setup for Logistic Regresion
1. Download and unzip https://bit.ly/2GE1Frw into Logistic_Regression/Glove/
2. Unzip Data.zip in Logistic_Regression/
3. From Logistic_Regression, run python3 run.py


## Setup for BERT

### To evaluate the model:

1. Download and unzip the trained model from https://bit.ly/2XDOe0k to a folder called large_models/
2. run python3 evaluate_bert.py

### To train the model

1. Run python3 race_bert.py it will run your model and save it in a folder called large_models/
2. Run evaluate_bert.py post that to run it in test set.

The folder Hypo contains all the hypothesis we tested, details regarding those are in the slides, technical report and the blog post

