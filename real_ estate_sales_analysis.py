import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split

# --- Module 1: Importing Data Sets ---
file_name = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/FinalModule_Coursera/data/kc_house_data_NaN.csv'
df = pd.read_csv(file_name)

# Displaying basic data info
print("Initial Dataframe Head:")
print(df.head())
print("\nData Types:")
print(df.dtypes)

# --- Module 2: Data Wrangling ---
# Dropping unnecessary columns
df.drop(columns=["id", "Unnamed: 0"], inplace=True)
print("\nSummary Statistics after dropping columns:")
print(df.describe())

# Handling missing values for 'bedrooms' and 'bathrooms'
mean_bedrooms = df['bedrooms'].mean()
df['bedrooms'].replace(np.nan, mean_bedrooms, inplace=True)

mean_bathrooms = df['bathrooms'].mean()
df['bathrooms'].replace(np.nan, mean_bathrooms, inplace=True)

print(f"\nMissing values remaining: {df.isnull().sum().sum()}")

# --- Module 3: Exploratory Data Analysis ---
# Counting houses with unique floor values
floor_counts = df['floors'].value_counts().to_frame()
print("\nFloor counts:")
print(floor_counts)

# Determining if houses with a waterfront view have more price outliers
plt.figure(figsize=(8, 6))
sns.boxplot(x='waterfront', y='price', data=df)
plt.title("Price Distribution: Waterfront vs Non-Waterfront")
plt.show()

# Determining correlation of 'sqft_above' with 'price'
sns.regplot(x='sqft_above', y='price', data=df)
plt.title("Regression: Price vs Sqft Above")
plt.show()

# --- Module 4: Model Development ---
# Linear Regression with sqft_living
X = df[['sqft_living']]
Y = df['price']
lm = LinearRegression()
lm.fit(X, Y)
print(f"\nLinear Model R^2 (sqft_living): {lm.score(X, Y)}")

# Pipeline with Scaling, Polynomial Features, and Linear Regression
features = ["floors", "waterfront", "lat", "bedrooms", "sqft_basement", 
            "view", "bathrooms", "sqft_living15", "sqft_above", "grade", "sqft_living"]
X_multi = df[features]
Input = [('scale', StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model', LinearRegression())]
pipe = Pipeline(Input)
pipe.fit(X_multi, Y)
print(f"Pipeline Model R^2: {pipe.score(X_multi, Y)}")

# --- Module 5: Model Evaluation and Refinement ---
x_train, x_test, y_train, y_test = train_test_split(X_multi, Y, test_size=0.15, random_state=1)

# Ridge Regression
RidgeModel = Ridge(alpha=0.1)
RidgeModel.fit(x_train, y_train)
print(f"Ridge Model R^2 (Test Set): {RidgeModel.score(x_test, y_test)}")

# Polynomial Ridge Regression (Degree 2)
pr = PolynomialFeatures(degree=2)
x_train_pr = pr.fit_transform(x_train)
x_test_pr = pr.transform(x_test)
RidgeModel_PR = Ridge(alpha=0.1)
RidgeModel_PR.fit(x_train_pr, y_train)
print(f"Polynomial Ridge R^2 (Test Set): {RidgeModel_PR.score(x_test_pr, y_test)}")