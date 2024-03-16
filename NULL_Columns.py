import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class Plotter:
    @staticmethod
    def visualize_nulls(df):
        plt.figure(figsize=(10, 6))
        sns.heatmap(df.isnull(), cmap='viridis', cbar=False, yticklabels=False)
        plt.title('Missing Values in DataFrame')
        plt.show()

class DataFrameTransform:
    @staticmethod
    def check_nulls(df):
        return df.isnull().sum()

    @staticmethod
    def drop_columns(df, columns_to_drop):
        existing_columns = df.columns.intersection(columns_to_drop)
        df = df.drop(columns=existing_columns)
        return df
    
    @staticmethod
    def impute_missing(df, strategy='median'):
        numeric_columns = df.select_dtypes(include=['number']).columns
        non_numeric_columns = df.select_dtypes(exclude=['number']).columns

        if strategy == 'median':
            df_numeric = df[numeric_columns].fillna(df[numeric_columns].median())
        elif strategy == 'mean':
            df_numeric = df[numeric_columns].fillna(df[numeric_columns].mean())

        df_non_numeric = df[non_numeric_columns].fillna(df[non_numeric_columns].mode().iloc[0])

        return pd.concat([df_numeric, df_non_numeric], axis=1)


# Example usage:
# Assuming df is your DataFrame
# Step 1: Determine the amount of NULLs in each column
# Load DataFrame from CSV file
df = pd.read_csv('/Users/fahiyeyusuf/Desktop/CLIF_data.csv')

# Check for NULL values
null_counts = DataFrameTransform.check_nulls(df)
print("NULL counts in each column:")
print(null_counts)

# Step 2: Drop columns with a large amount of missing values
columns_to_drop = ['funded_amount', 'term', 'int_rate', 'employment_length', 'mths_since_last_deling', 'mths_since_last_record', 'last_payment_date', 'next_payment_date', 'last_credit_pull_date', 'collections_12_mths_ex_med', 'mths_since_last_major_derog']  # Adjust this list based on null_counts
df = DataFrameTransform.drop_columns(df, columns_to_drop)

# Step 3: Impute missing values
# Let's say we choose to impute missing values with the median for numerical columns
df = DataFrameTransform.impute_missing(df, strategy='median')

# Step 4: Check that all NULLs have been removed
null_counts_after = DataFrameTransform.check_nulls(df)
print("\nNULL counts in each column after imputation:")
print(null_counts_after)

# Visualize the removal of NULL values
Plotter.visualize_nulls(df)

# Save the cleaned DataFrame to a new CSV file
df.to_csv('/Users/fahiyeyusuf/Desktop/CLIF_data_null_cleaned.csv', index=False)
