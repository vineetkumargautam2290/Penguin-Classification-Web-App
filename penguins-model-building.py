!pip install ydata-profiling

import numpy as np
import pandas as pd

from pandas_profiling import ProfileReport

df_raw = pd.read_csv("/content/drive/MyDrive/streamlit development /Penguin Classification Web App/penguins.csv")
df = df_raw.copy()
df.head()

df.info()

df.isnull().sum()

def impute_missing_values(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            # For non-numeric columns, fill missing values with mode
            df[col].fillna(df[col].mode().iloc[0], inplace=True)
        else:
            # For numeric columns, fill missing values with median
            df[col].fillna(df[col].median(), inplace=True)
    return df

df = impute_missing_values(df)

df.isnull().sum()

df.drop(['Unnamed: 0','year'], inplace=True,axis=1)

df.head()

df.describe(include='all')

profile = ProfileReport(df, title='Pandas Profiling Report', explorative=True)

# Step 6: Save the report to a file
profile.to_file("pandas_profiling_report.html")

# Optional: Display the report in the notebook
profile.to_notebook_iframe()

df.species = df.species.map({'Adelie':0, 'Chinstrap':1, 'Gentoo':2})

df = df[['island', 'sex', 'bill_length_mm', 'bill_depth_mm','flipper_length_mm', 'body_mass_g','species']]

df.head()

df_new = pd.get_dummies(df, columns=['island','sex'], drop_first=True, dtype='int')

df_new.head()

# Separating X and y
X = df_new.drop('species', axis=1)
Y = df_new['species']

X.head()

X.info()

Y.head()

Y.value_counts(normalize=True)

from sklearn.model_selection import train_test_split

# Assuming 'X' contains your features and 'y' contains your target variable

# Split the data into training and testing sets while stratifying by 'y'
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)

# Print the shapes of the resulting datasets
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb

def fit_and_evaluate(X_train, X_test, y_train, y_test):

    # Step 2: Instantiate classifiers
    classifiers = {
        'Logistic Regression': LogisticRegression(),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'Support Vector Machine': SVC(),
        'XGBoost': xgb.XGBClassifier()
    }

    # Step 3: Fit and evaluate each classifier
    results = {}
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)

        # Classification report
        clf_report = classification_report(y_test, y_pred)

        # Precision, recall, fscore
        precision, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

        results[name] = {
            'Accuracy': accuracy,
            'Classification Report': clf_report,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': fscore
        }

    return results

# Example usage:
# Assuming X contains your features and y contains your target variable
results = fit_and_evaluate(X_train, X_test, y_train, y_test)

# Print the results
for clf, metrics in results.items():
    print(f"Classifier: {clf}")
    print(f"Accuracy: {metrics['Accuracy']:.4f}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall: {metrics['Recall']:.4f}")
    print(f"F1 Score: {metrics['F1 Score']:.4f}")
    print("Classification Report:")
    print(metrics['Classification Report'])
    print("\n")

"""Random forest and XGBOOST gave the best result."""

model = xgb.XGBClassifier()

# Train the model
model.fit(X_train, y_train)

# Saving the model
import pickle
pickle.dump(model, open('penguins_clf.pkl', 'wb'))