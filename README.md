# EXPERIMENT 06: HEART ATTACK PREDICTION USING MLP
## AIM:
To construct a  Multi-Layer Perceptron to predict heart attack using Python.
## ALGORITHM:
Step 1:Import the required libraries: numpy, pandas, MLPClassifier, train_test_split, StandardScaler, accuracy_score, and matplotlib.pyplot.<br>
Step 2:Load the heart disease dataset from a file using pd.read_csv().<br>
Step 3:Separate the features and labels from the dataset using data.iloc values for features (X) and data.iloc[:, -1].values for labels (y).<br>
Step 4:Split the dataset into training and testing sets using train_test_split().<br>
Step 5:Normalize the feature data using StandardScaler() to scale the features to have zero mean and unit variance.<br>
Step 6:Create an MLPClassifier model with desired architecture and hyperparameters, such as hidden_layer_sizes, max_iter, and random_state.<br>
Step 7:Train the MLP model on the training data using mlp.fit(X_train, y_train). The model adjusts its weights and biases iteratively to minimize the training loss.<br>
Step 8:Make predictions on the testing set using mlp.predict(X_test).<br>
Step 9:Evaluate the model's accuracy by comparing the predicted labels (y_pred) with the actual labels (y_test) using accuracy_score().<br>
Step 10:Print the accuracy of the model.<br>
Step 11:Plot the error convergence during training using plt.plot() and plt.show().<br>

## PROGRAM:
```PYTHON

import numpy as np
import pandas as pd 
from sklearn.neural_network import MLPClassifier 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

data=pd.read_csv("/content/heart.csv")
X=data.iloc[:, :-1].values #features 
Y=data.iloc[:, -1].values  #labels 

X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

mlp=MLPClassifier(hidden_layer_sizes=(100,100),max_iter=1000,random_state=42)
training_loss=mlp.fit(X_train,y_train).loss_curve_

y_pred=mlp.predict(X_test)

accuracy=accuracy_score(y_test,y_pred)
print("Accuracy",accuracy)

plt.plot(training_loss)
plt.title("MLP Training Loss Convergence")
plt.xlabel("Iteration")
plt.ylabel("Training Losss")
plt.show()

```

## OUTPUT:
### X Values:
![image](https://github.com/Rithigasri/Experiment-6---Heart-attack-prediction-using-MLP/assets/93427256/7874236c-c1c6-42db-afb5-6494d92e55aa)
### Y Values:
![image](https://github.com/Rithigasri/Experiment-6---Heart-attack-prediction-using-MLP/assets/93427256/fc0f7ac8-ea42-4697-8a45-3f55d3e2dc4a)
### Accuracy:
![image](https://github.com/Rithigasri/Experiment-6---Heart-attack-prediction-using-MLP/assets/93427256/a8c78ea2-2729-4794-9636-edacd69580c2)
### Loss Convergence Graph:
![image](https://github.com/Rithigasri/Experiment-6---Heart-attack-prediction-using-MLP/assets/93427256/b2df2007-899b-45b0-91a7-224e036f46dc)

## RESULT:
Thus, an ANN with MLP is constructed and trained to predict the heart attack using python.
     

