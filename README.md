# Customer Churn Prediction using Deep Learning

## Overview

This project is a deep learning-based classification model designed to predict customer churn. The model uses a neural network to analyze customer data and determine whether a customer is likely to leave the service (churn) or not. The project is implemented using TensorFlow, Pandas, and Scikit-learn, and it includes data preprocessing, model training, and prediction capabilities.

## Project Structure

The project consists of the following files:

1. **experiments.ipynb**: This Jupyter notebook contains the code for data preprocessing, model training, and evaluation. It also includes TensorBoard integration for visualizing training metrics.
2. **prediction.ipynb**: This notebook demonstrates how to load the trained model and use it to make predictions on new data.
3. **requirements.txt**: This file lists all the Python packages required to run the project.

## Installation

To set up the project, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

## Data Preprocessing

The dataset used in this project is `Churn_Modelling.csv`, which contains customer information such as credit score, geography, gender, age, tenure, balance, and more. The preprocessing steps include:

- **Dropping irrelevant columns**: Columns like `RowNumber`, `CustomerId`, and `Surname` are removed as they do not contribute to the prediction.
- **Encoding categorical variables**: The `Gender` column is label encoded, and the `Geography` column is one-hot encoded.
- **Scaling features**: The numerical features are scaled using `StandardScaler` to ensure that all features contribute equally to the model.

## Model Training

The model is a simple feedforward neural network built using TensorFlow's Keras API. The architecture of the model is as follows:

- **Input Layer**: Takes in the preprocessed features.
- **Hidden Layers**: Two dense layers with ReLU activation functions.
- **Output Layer**: A single neuron with a sigmoid activation function to output the probability of churn.

The model is trained using the `Adam` optimizer and `BinaryCrossentropy` loss function. Early stopping is implemented to prevent overfitting, and TensorBoard is used to monitor training progress.

## Making Predictions

The `prediction.ipynb` notebook demonstrates how to load the trained model and use it to make predictions on new data. The steps include:

1. **Loading the model and encoders**: The trained model, one-hot encoder, label encoder, and scaler are loaded from saved files.
2. **Preprocessing new data**: The input data is preprocessed in the same way as the training data (one-hot encoding, label encoding, and scaling).
3. **Making predictions**: The model predicts the probability of churn for the new data. If the probability is greater than 0.5, the customer is classified as likely to churn.

## Usage

To train the model and make predictions, follow these steps:

1. **Train the model**:
   - Open `experiments.ipynb` and run all cells to preprocess the data, train the model, and save the trained model and encoders.

2. **Make predictions**:
   - Open `prediction.ipynb` and run all cells to load the model and make predictions on new data.

## Dependencies

The project requires the following Python packages:

- TensorFlow (2.18.0)
- Pandas
- NumPy
- Scikit-learn
- TensorBoard
- Matplotlib
- Streamlit

These can be installed using the `requirements.txt` file.

## Conclusion

This project demonstrates how to build and deploy a deep learning model for customer churn prediction. The model is trained on customer data and can be used to predict whether a customer is likely to churn, helping businesses take proactive measures to retain customers.

For any questions or issues, please refer to the project documentation or contact the project maintainer.

---

**Note**: Ensure that you have the necessary data (`Churn_Modelling.csv`) in the project directory before running the notebooks.
