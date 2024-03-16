import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load the dataset and preprocess it
df = pd.read_csv('/Users/fahiyeyusuf/Desktop/CLIF_data.csv')

# Exclude specific columns from the removal process
columns_to_exclude = ['id', 'loan_amount', 'total_payment', 'recoveries']

# Drop non-numeric columns or convert them to numeric if possible
numeric_columns = df.select_dtypes(include=[np.number]).columns
df_numeric = df[numeric_columns]

# Compute the correlation matrix
correlation_matrix = df_numeric.corr()

# Step 2: Visualize the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# Step 3: Identify highly correlated columns
correlation_threshold = 0.7  # Define your correlation threshold here
highly_correlated_columns = set()
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > correlation_threshold:
            colname_i = correlation_matrix.columns[i]
            colname_j = correlation_matrix.columns[j]
            if colname_i not in columns_to_exclude and colname_j not in columns_to_exclude:
                highly_correlated_columns.add(colname_i)
                highly_correlated_columns.add(colname_j)

# Step 4: Decide which columns to remove
columns_to_remove = list(highly_correlated_columns)
print("Columns highly correlated and to be removed:", columns_to_remove)

# Step 5: Remove highly correlated columns from the dataset
df_cleaned = df.drop(columns=columns_to_remove)

# Optionally, you can save the cleaned dataset to a new CSV file
df_cleaned.to_csv('/Users/fahiyeyusuf/Desktop/CLIF_data_cleaned.csv', index=False)
