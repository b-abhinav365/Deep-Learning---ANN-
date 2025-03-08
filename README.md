# Deep-Learning---ANN-
Project Title: Bank Customer Churn Prediction using ANN

# Customer Churn Prediction using Artificial Neural Networks (ANN)

## Project Overview

This project predicts customer churn for a bank using an Artificial Neural Network (ANN). The dataset contains customer details, and the goal is to classify whether a customer will leave the bank or not.

## What is an Artificial Neural Network (ANN)?

An ANN is a computational model inspired by the human brain. It consists of layers of neurons that process inputs, learn patterns, and make predictions. The main components of an ANN include:

- **Input Layer**: Receives input features.
- **Hidden Layers**: Learn patterns through weighted connections.
- **Output Layer**: Produces the final prediction.

## Step-by-Step Code Explanation

### 1. Importing Required Libraries

```python
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
```

- `pandas` and `numpy`: Handle and process data.
- `tensorflow` and `keras`: Build and train the ANN model.
- `sklearn.model_selection`: Split the dataset.
- `StandardScaler`: Standardize numerical data.
- `LabelEncoder`: Convert categorical labels into numerical form.

### 2. Loading the Dataset

```python
df = pd.read_csv('customer_churn.csv')
```

- Loads customer churn data into a DataFrame.

### 3. Data Preprocessing

```python
X = df.drop(columns=['Exited'])  # Features
y = df['Exited']  # Target variable
```

- `X` contains all features except the target column `Exited`.
- `y` stores the churn labels (0: Not Churned, 1: Churned).

```python
X = pd.get_dummies(X, drop_first=True)
```

- Converts categorical variables into numerical format using one-hot encoding.

```python
scaler = StandardScaler()
X = scaler.fit_transform(X)
```

- Standardizes features for better ANN performance.

### 4. Splitting Data into Training and Test Sets

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

- Splits the data into 80% training and 20% testing.

### 5. Building the ANN Model

```python
model = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
```

- **First Hidden Layer**: 16 neurons with ReLU activation.
- **Second Hidden Layer**: 8 neurons with ReLU activation.
- **Output Layer**: 1 neuron with Sigmoid activation (for binary classification).

### 6. Compiling and Training the Model

```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

- `adam`: Optimizer that adapts learning rate.
- `binary_crossentropy`: Suitable loss function for binary classification.
- `accuracy`: Metric to evaluate performance.

```python
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
```

- Trains the model for 50 epochs using a batch size of 32.
- Validates performance on test data after each epoch.

### 7. Evaluating the Model

```python
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.2f}")
```

- Evaluates model performance on test data.
- Prints test accuracy.

## Conclusion

This ANN model effectively predicts customer churn based on historical data. Feature engineering, standardization, and a well-structured neural network are key to improving accuracy.




