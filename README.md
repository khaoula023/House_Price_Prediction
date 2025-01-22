# House_Price_Prediction

This project implements a house price prediction model using ensemble learning techniques, specifically Bagging with multiple Support Vector Regression (SVR) models. The objective is to create a robust and accurate model to predict house prices based on various features.

## Features

The key features of the project include:

- **Ensemble Learning**: Combines multiple SVR models to improve prediction accuracy and reduce overfitting.

- **Bagging Technique**: Uses bootstrap aggregation to train multiple SVR models on different subsets of the dataset.

- **Support Vector Regression**: Employs SVR, a powerful regression technique, for each base model.

- **Scalable Architecture**: Allows for customization of hyperparameters and scalability to larger datasets.

## Dataset

The project uses a house pricing dataset, which contains the following features:

- **Numerical features**: (e.g., Area, number of Bedrooms, Year Built)

- **Categorical features**: (e.g., Location, Garage)

- **Target variable**: House price
  The dataset is preprocessed to handle missing values, encode categorical features, and normalize numerical data.

## Methodology

### 1.Data Preprocessing:

- Handle missing values.

- Encode categorical variables using one-hot encoding.

- Normalize numerical features to ensure all features contribute equally.

### 2.Model Building:

- Create multiple SVR models with different hyperparameters.

- Use the Bagging technique to train the models on bootstrapped datasets.

### 3.Model Evaluation:

- Evaluate the ensemble model using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared.

### 4.Create Flask App for Real-Time Prediction:

- Develop a Flask application to serve the trained ensemble model.

- Provide an API endpoint where users can send input features and receive predicted house prices in real-time.

## Tools and Libraries

- **Python**

- **Scikit-learn**: For machine learning models and preprocessing.

- **Pandas**: For data manipulation and analysis.

- **NumPy**: For numerical operations.

- **Matplotlib/Seaborn**: For data visualization
- **Flask**: For deploying the model as a web application
