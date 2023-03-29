Housing Price Prediction


This Python script reads in a housing dataset from a CSV file, performs some basic data cleaning and preprocessing steps, and trains two regression models to predict housing prices. The script uses the scikit-learn library to perform the following tasks:


Data set: https://www.kaggle.com/datasets/camnugent/california-housing-prices

Split the data into training and testing sets
Perform some feature engineering, including log-transforming some variables and creating new variables based on existing ones
Train a linear regression model and a random forest regressor model on the training data
Use grid search cross-validation to optimize the hyperparameters of the random forest model
Compute the performance of both models on the test set


Here's a brief description of each block of code:

Importing necessary libraries: The code starts by importing required libraries like pandas, numpy, matplotlib, seaborn, and scikit-learn. These libraries are essential for data analysis, data visualization, and machine learning tasks.

Reading data: The housing data is read from a CSV file named "housing.csv" and stored in a Pandas DataFrame.

Data cleaning: The code checks if there are any null values or invalid data in the dataset and drops rows with null values.

![image](https://user-images.githubusercontent.com/82034400/228428205-c100fe51-a19c-4ff6-8c54-17fc8ca1ff89.png)

Data preparation: The dataset is split into training and testing sets using the train_test_split function from scikit-learn. The target variable "median_house_value" is separated from the input features. The histograms of the training data are plotted, and a heatmap of the correlation matrix is generated using seaborn.

Feature engineering: Four features are transformed using the logarithm function to reduce their skewness. A scatter plot of latitude and longitude is generated using the median_house_value as the color. Two new features "bedroom_ratio" and "household_rooms" are added to the dataset. One-hot encoding is applied to the categorical feature "ocean_proximity."

Linear Regression: A Linear Regression model is trained on the training set and evaluated on the testing set using the R2 score.

Random Forest Regression: A Random Forest Regression model is trained on the training set and evaluated on the testing set using the R2 score.

Hyperparameter Tuning: A GridSearchCV is used to search for the best hyperparameters for the Random Forest Regression model. The best model is then evaluated on the testing set using the R2 score.

Outputs of some Heatmaps, scatterplots and accuracy scores:

        Heatmaps:

        ![image](https://user-images.githubusercontent.com/82034400/228428325-162b2d34-9545-4b4b-b44b-94860d0d2859.png)

        ![image](https://user-images.githubusercontent.com/82034400/228428379-0db744fd-4392-4c41-804b-98f6381dc3b4.png)

        ![image](https://user-images.githubusercontent.com/82034400/228428268-7796f972-27f5-4f5c-8155-4fe1f64ef665.png)

        Accuracy scores:
        
        ![image](https://user-images.githubusercontent.com/82034400/228428425-d1994e71-4b81-438c-8c94-39e8c6d1a4f0.png)

        ![image](https://user-images.githubusercontent.com/82034400/228428453-d4502509-4d02-4d61-bd48-453da4ae57d3.png)



Dependencies

This script requires the following Python libraries:

pandas
numpy
matplotlib
seaborn
scikit-learn
Usage
To use this script, follow these steps:

Download the housing.csv file from the California Housing Prices dataset and save it in the same directory as the script.
Run the script in a Python environment that has the required libraries installed.
The script will output some plots and print out the performance of the two models on the test set.
