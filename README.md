This project is going to measure Nitrogen oxide emission of gas turbine by using NOX as target variable.

## Phase 1: EDA and Data Pre-processing

**Preprocess and Cleaning**

- Checking for any irrelevant features in the dataset and dropping them.

- Checking for null / missing values.

- Check for different data types.

**Exploratory Data Analysis through Visualisation**

 Then use various graphs to understand the data better.

 - Histplot
   
 - Pairplot
   
 - Scatter Plot

 - Boxplot (Handle outliers)

 - Heatmap

## Phase 2: Feature Selection and Model Building

Based on model evaluation metrics, Random Forest outperformed the other models, achieving the best overall predictive performance, indicating it is the most suitable model for this dataset.

# Model Output Preview

![Model Output](output.png)

![Model Output](output1.png)

![Model Output](output2.png)

## *NOTE:*
Trained model files (.pkl) are excluded due to size limits.
Run model.py to regenerate the model.

## Requirements

- Python 3.10+

- Flask

- Pandas

- NumPy

- Scikit-learn

- Random Forest Regression Model

- HTML,CSS
