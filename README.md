# Cardiovascular Disease Prediction using Decision Trees

## Description
This Python script utilizes a decision tree classifier from scikit-learn to predict the presence of cardiovascular disease based on various health-related features. It preprocesses the data, trains the classifier, evaluates its performance, and provides insights into the predictive power of different features.

## Dependencies
- Python 3.x
- numpy
- pandas
- matplotlib
- scikit-learn
- pydotplus


## Usage
1. Place your dataset file named `cardio_train.csv` in the same directory as the script.
2. Run the script `cardio_decision_tree.py`.
3. The script will preprocess the data, split it into training and testing sets, train the decision tree classifier, evaluate its performance using accuracy score and classification report, and generate visualizations of the decision tree.
4. The decision tree visualization will be saved as `main_tree.png`.

## Features Information
- **Age:** 1 represents individuals over 50 years old, 0 represents individuals under 50.
- **Gender:** 1 represents female, 0 represents male.
- **Systolic Blood Pressure (SBP):** 1 represents SBP above 140, 0 represents SBP under 140.
- **Diastolic Blood Pressure (DBP):** 1 represents DBP above 90, 0 represents DBP under 90.
- **Cholesterol Level:** 1 represents cholesterol level above normal, 0 represents normal.
- **Glucose Level:** 1 represents glucose level above normal, 0 represents normal.
- **Smoking:** 1 represents smoking, 0 represents not smoking.
- **Alcohol intake:** 1 represents drinking, 0 represents not drinking.
- **Physical activity:** 1 represents being active, 0 represents not being active.

## File Description
- `cardio_decision_tree.py`: The Python script containing the decision tree classifier implementation.
- `cardio_train.csv`: Dataset containing health-related features for predicting cardiovascular disease.

