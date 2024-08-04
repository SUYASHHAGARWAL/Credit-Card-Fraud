# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA

# Load the dataset
def load_data(filepath):
    try:
        df = pd.read_csv(filepath, delimiter=',', quotechar='"', on_bad_lines='skip')
        return df
    except pd.errors.ParserError as e:
        print(f"Error loading data: {e}")
        return None



# Data Cleaning
def clean_data(df):
    # Convert all columns to numeric if possible
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # Handle missing values
    df.fillna(df.median(), inplace=True)

    # Convert categorical features to numerical if any (Example given for 'Category')
    if 'Category' in df.columns:
        df = pd.get_dummies(df, columns=['Category'], drop_first=True)

    return df

# Data Normalization/Standardization
def normalize_data(df, columns_to_normalize):
    scaler = StandardScaler()
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize].astype(float))
    return df

# Data Splitting
def split_data(df, target_column, test_size=0.2, random_state=42):
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

# Handle imbalanced data using SMOTE
def handle_imbalanced_data(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    return X_train_res, y_train_res

# Feature selection using PCA
def apply_pca(X_train, X_test, n_components=10):
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca

# Visualize the class distribution
def plot_class_distribution(y):
    sns.countplot(x=y)
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.show()

# Visualize the correlation matrix
def plot_correlation_matrix(df):
    plt.figure(figsize=(12, 8))
    corr = df.corr()
    sns.heatmap(corr, cmap='coolwarm', annot=False)
    plt.title('Correlation Matrix')
    plt.show()

# Plot histograms for the features
def plot_feature_histograms(df):
    df.hist(figsize=(20, 20), bins=50)
    plt.show()

# Check for imbalanced data
def check_imbalanced_data(y):
    fraud_count = y.value_counts()[1]
    normal_count = y.value_counts()[0]
    print(f"Number of fraudulent transactions: {fraud_count}")
    print(f"Number of normal transactions: {normal_count}")

# Train and evaluate a Random Forest model
def train_evaluate_model(X_train, X_test, y_train, y_test):
    # Initialize the Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Train the model
    rf.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = rf.predict(X_test)
    
    # Evaluate the model
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    # Path to the dataset
    filepath = 'creditcard.csv'

    # Load the dataset
    df = load_data(filepath)
    print("Dataset Loaded:")
    print(df.head())

    # Clean the data
    df = clean_data(df)
    print("Data Cleaned:")
    print(df.head())

    # Columns to normalize (add more if needed)
    columns_to_normalize = ['Amount']

    # Normalize the data
    df = normalize_data(df, columns_to_normalize)
    print("Data Normalized:")
    print(df.head())

    # Split the data
    X_train, X_test, y_train, y_test = split_data(df, target_column='Class')
    print(f"Training set size: {X_train.shape}")
    print(f"Testing set size: {X_test.shape}")

    # Perform EDA
    # Extract the target variable
    y = df['Class']

    # Plot class distribution
    plot_class_distribution(y)

    # Plot correlation matrix
    plot_correlation_matrix(df)

    # Plot histograms for features
    plot_feature_histograms(df)

    # Check for imbalanced data
    check_imbalanced_data(y)

    # Handle imbalanced data
    X_train_res, y_train_res = handle_imbalanced_data(X_train, y_train)
    print(f"Resampled training set size: {X_train_res.shape}")

    # Apply PCA for feature selection
    X_train_pca, X_test_pca = apply_pca(X_train_res, X_test)
    print(f"PCA training set size: {X_train_pca.shape}")
    print(f"PCA testing set size: {X_test_pca.shape}")

    # Train and evaluate the model
    train_evaluate_model(X_train_pca, X_test_pca, y_train_res, y_test)
