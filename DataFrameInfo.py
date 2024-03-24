import pandas as pd

class DataFrameInfo:
    def __init__(self, df):
        self.df = df

    def describe_columns(self):
        return self.df.dtypes

    def extract_statistics(self):
        return self.df.describe()

    def count_distinct_values(self):
        distinct_counts = {}
        for column in self.df.columns:
            if self.df[column].dtype == 'object':
                distinct_counts[column] = len(self.df[column].unique())
        return distinct_counts

    def print_shape(self):
        print("Shape of DataFrame:", self.df.shape)

    def count_null_values(self):
        return self.df.isnull().sum()

# Load DataFrame from CSV file
df = pd.read_csv('/Users/fahiyeyusuf/Desktop/CLIF_data.csv')

# Example usage:
df_info = DataFrameInfo(df)

# Describe all columns
print("Data Types of Columns:")
print(df_info.describe_columns())

# Extract statistical values
print("\nStatistical Values:")
print(df_info.extract_statistics())

# Count distinct values in categorical columns
print("\nDistinct Value Counts:")
print(df_info.count_distinct_values())

# Print out the shape of the DataFrame
print("\nDataFrame Shape:")
df_info.print_shape()

# Generate a count/percentage count of NULL values in each column
print("\nCount of NULL Values:")
print(df_info.count_null_values())
