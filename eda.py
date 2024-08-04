# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

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

if __name__ == "__main__":
    # Path to the dataset
    filepath = 'creditcard.csv'

    # Load the dataset
    df = load_data(filepath)

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
