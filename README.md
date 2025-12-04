"# Customer-Churn-Prediction-DL" 
# Customer Churn Prediction using Neural Networks

This project predicts customer churn using a neural network implemented in TensorFlow/Keras. The dataset used is the "Churn_Modelling.csv" dataset. The goal is to classify which customers are likely to leave a company based on their profile and behavior.


## Dataset
The dataset contains customer information such as:
- Credit Score
- Geography
- Gender
- Age
- Tenure
- Balance
- Number of Products
- Has Credit Card
- Is Active Member
- Estimated Salary
- Exited (Target variable)

The dataset is available as `Churn_Modelling.csv`.

## Features
- Dropped unnecessary columns: `CustomerId`, `RowNumber`, `Surname`.
- Encoded categorical features (`Geography`, `Gender`) using one-hot encoding.
- Scaled numerical features using `StandardScaler`.

## Model
- Neural network with 2 hidden layers (16 neurons each, ReLU activation)
- Output layer: 1 neuron, Sigmoid activation
- Optimizer: Adam
- Loss: Binary Crossentropy
- Metrics: Accuracy
- Trained for 100 epochs with batch size 16 and 20% validation split

## Installation
1. Clone this repository:
git clone https://github.com/hamza-amjad10/Customer-Churn-Prediction-DL.git


Install dependencies:

pip install pandas numpy scikit-learn tensorflow matplotlib

Run the script:

python model.py