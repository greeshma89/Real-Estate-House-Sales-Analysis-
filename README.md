# House Sales Analysis in King County, USA

This project explores and predicts house sale prices in King County, Washington (including Seattle) using data collected between May 2014 and May 2015. 

## Project Overview
The goal of this analysis is to perform Exploratory Data Analysis (EDA) and develop various regression models to predict house prices based on features like square footage, number of floors, and waterfront views.

## Dataset Features
The dataset includes several variables for each home, such as:
- **Price**: Prediction target.
- **Bedrooms/Bathrooms**: Number of rooms.
- **Sqft_living**: Square footage of the home.
- **Waterfront**: View to a waterfront.
- **Grade**: Overall grade based on the King County grading system.

## Key Modules
1. **Data Importing**: Loading the dataset from a remote CSV.
2. **Data Wrangling**: Cleaning the data by dropping identifiers and imputing missing values with column means.
3. **Exploratory Data Analysis (EDA)**: Using visualization tools like Seaborn and Matplotlib to find correlations and outliers.
4. **Model Development**:
   - Simple Linear Regression.
   - Multi-feature Pipelines with `StandardScaler` and `PolynomialFeatures`.
5. **Model Evaluation**: Using Ridge Regression and Train/Test splits to refine accuracy and prevent overfitting.

## Requirements
To run this script, you will need the following libraries:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

Install them using pip:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
