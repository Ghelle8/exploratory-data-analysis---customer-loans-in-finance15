# Exploratory Data Analysis - Customer Loans in Finance

## Table of Contents

1. [Description](#description)
2. [Installation](#installation)
3. [Usage](#usage)
4. [File Structure](#file-structure)
5. [License](#license)

## Description

This project aims to perform exploratory data analysis on customer loans in the finance domain. It includes functions to load data from a CSV file into a Pandas DataFrame and extract data from an RDS database, as well as save data to a CSV file. Through this project, I've learned how to work with data from financial databases, connect to databases using SQLAlchemy, and perform data analysis using Pandas.

## Installation

To install the required dependencies, run:

pip install pandas sqlalchemy


## Usage

1. **Loading Data**: Use the `load_data` function to load data from a CSV file into a Pandas DataFrame.

```python
import pandas as pd

def load_data(file_path):
    # Implementation here...

# Example usage:
file_path = 'data.csv'
data_df = load_data(file_path)
```
2. **Connecting to RDS Database**: Create an instance of RDSDatabaseConnector with your RDS credentials and use the connect method to establish a connection.

```python
from sqlalchemy import create_engine

class RDSDatabaseConnector:
    # Implementation here...

# Example usage:
credentials = {
    'RDS_HOST': 'eda-projects.cq2e8zno855e.eu-west-1.rds.amazonaws.com',
    'RDS_PASSWORD': 'EDAloananalyst',
    'RDS_USER': 'loansanalyst',
    'RDS_DATABASE': 'payments',
    'RDS_PORT': 5432
}

connector = RDSDatabaseConnector(credentials)
connector.connect()
```
3. **Extracting Data from RDS Database**: Use the extract_data method to fetch data from the RDS database.
```python
data_df = connector.extract_data()
```
4. **Saving Data to CSV**: Use the save_to_csv function to save a DataFrame to a CSV file
```python
def save_to_csv(dataframe, file_path):
    # Implementation here...

# Example usage:
file_path = 'data.csv'
save_to_csv(data_df, file_path)
```
5. **Data Transformation**: Use the DataFrameTransformer class to perform various data transformations.
```python
import pandas as pd

class DataFrameTransformer:
    @staticmethod
    def _apply_transformation(df, columns, transformation_func):
        for col in columns:
            df[col] = transformation_func(df[col])
        return df

    @staticmethod
    def convert_columns_to_numeric(df, columns):
        transformation_func = pd.to_numeric
        return DataFrameTransformer._apply_transformation(df, columns, transformation_func)

    @staticmethod
    def convert_columns_to_datetime(df, columns, date_formats=None):
        if date_formats is None:
            date_formats = {}
        
        def transform_date(col):
            if col.name in date_formats:
                return pd.to_datetime(col, errors='coerce', format=date_formats[col.name])
            else:
                return pd.to_datetime(col, errors='coerce')

        return DataFrameTransformer._apply_transformation(df, columns, transform_date)

    @staticmethod
    def convert_columns_to_categorical(df, columns):
        transformation_func = lambda col: col.astype('category')
        return DataFrameTransformer._apply_transformation(df, columns, transformation_func)

    @staticmethod
    def remove_excess_symbols(df, columns, symbols):
        transformation_func = lambda col: col.str.replace(symbols, '')
        return DataFrameTransformer._apply_transformation(df, columns, transformation_func)

if __name__ == "__main__":
    # Load data
    df = pd.read_csv('data.csv')

    # Instantiate DataFrameTransformer class
    transformer = DataFrameTransformer()

    # Define columns needing format adjustments
    numeric_columns = ['loan_amount', 'funded_amount', 'funded_amount_inv', 'annual_inc', 'last_payment_amount', 'open_accounts', 'total_accounts', 'out_prncp', 'out_prncp_inv', 'total_payment', 'total_payment_inv', 'total_rec_prncp', 'total_rec_int' ]  # Update with actual column names
    date_columns = ['issue_date', 'last_payment_date', 'next_payment_date', 'last_credit_pull_date' ]  # Update with actual column names
    categorical_columns = ['grade', 'sub_grade', 'home_ownership', 'verification_status', 'loan_status', 'application_type', 'payment_plan', 'grade', 'sub_grade' ]  # Update with actual column names
    columns_with_symbols = ['purpose']  # Update with actual column names

    # Specify date formats for specific columns
    date_formats = {'issue_date': '%Y-%m-%d', 'last_payment_date': '%Y/%m/%d', 'next_payment_date': '%d-%m-%Y', 'last_credit_pull_date': '%m/%d/%Y'}

    # Apply transformations
    df = transformer.convert_columns_to_numeric(df, numeric_columns)
    df = transformer.convert_columns_to_datetime(df, date_columns, date_formats)
    df = transformer.convert_columns_to_categorical(df, categorical_columns)
    df = transformer.remove_excess_symbols(df, columns_with_symbols, '@#$')

    # Verify transformations
    print(df.dtypes)  # Check data types of columns
    print(df.head())  # View sample of transformed data
```
6. **Analysing DataFrame**: Use the DataFrameInfo class to analyse the DataFrame.
```python
import pandas as pd

class DataFrameInfo:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def _get_categorical_columns(self) -> list:
        return [col for col in self.df.columns if self.df[col].dtype == 'object']

    def _get_numeric_columns(self) -> list:
        return [col for col in self.df.columns if self.df[col].dtype != 'object']

    def describe_columns(self) -> pd.Series:
        return self.df.dtypes

    def get_statistics(self) -> pd.DataFrame:
        return self.df.describe()

    def count_distinct_values(self) -> dict:
        distinct_counts = {}
        for column in self._get_categorical_columns():
            distinct_counts[column] = len(self.df[column].unique())
        return distinct_counts

    def print_shape(self) -> None:
        print("Shape of DataFrame:", self.df.shape)

    def count_null_values(self) -> pd.Series:
        return self.df.isnull().sum()

if __name__ == "__main__":
    # Load DataFrame from CSV file
    file_path = '/Users/fahiyeyusuf/Desktop/CLIF_data.csv'
    df = pd.read_csv(file_path)

    # Example usage:
    df_info = DataFrameInfo(df)

    # Describe all columns
    print("Data Types of Columns:")
    print(df_info.describe_columns())

    # Extract statistical values
    print("\nStatistical Values:")
    print(df_info.get_statistics())

    # Count distinct values in categorical columns
    print("\nDistinct Value Counts:")
    print(df_info.count_distinct_values())

    # Print out the shape of the DataFrame
    print("\nDataFrame Shape:")
    df_info.print_shape()

    # Generate a count/percentage count of NULL values in each column
    print("\nCount of NULL Values:")
    print(df_info.count_null_values())
```
7. **Data Cleaning and Preprocessing**: Use the DataFrameTransform class to clean and preprocess data.
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class Plotter:
    @staticmethod
    def visualize_nulls(df: pd.DataFrame) -> None:
        plt.figure(figsize=(10, 6))
        sns.heatmap(df.isnull(), cmap='viridis', cbar=False, yticklabels=False)
        plt.title('Missing Values in DataFrame')
        plt.show()

class DataFrameTransform:
    @staticmethod
    def check_nulls(df: pd.DataFrame) -> pd.Series:
        return df.isnull().sum()

    @staticmethod
    def remove_specific_values(df: pd.DataFrame, columns_to_clean: dict) -> pd.DataFrame:
        """
        Remove specific values within columns and replace them with NaN.
        columns_to_clean should be a dictionary where keys are column names and values are lists of values to remove.
        """
        for col, values_to_remove in columns_to_clean.items():
            df[col] = df[col].replace(values_to_remove, pd.NA)
        return df

    @staticmethod
    def impute_missing(df: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
        numeric_columns = df.select_dtypes(include=['number']).columns
        non_numeric_columns = df.select_dtypes(exclude=['number']).columns

        if strategy == 'median':
            df_numeric = df[numeric_columns].fillna(df[numeric_columns].median())
        elif strategy == 'mean':
            df_numeric = df[numeric_columns].fillna(df[numeric_columns].mean())

        df_non_numeric = df[non_numeric_columns].fillna(df[non_numeric_columns].mode().iloc[0])

        return pd.concat([df_numeric, df_non_numeric], axis=1)

if __name__ == "__main__":
    # Load DataFrame from CSV file
    file_path = '/Users/fahiyeyusuf/Desktop/CLIF_data.csv'
    df = pd.read_csv(file_path)

    # Example usage:
    # Step 1: Determine the amount of NULLs in each column
    null_counts = DataFrameTransform.check_nulls(df)
    print("NULL counts in each column:")
    print(null_counts)

    # Step 2: Remove specific values within columns
    columns_to_clean = {'term': ['< 1 year', '10+ years'], 'employment_length': ['< 1 year', '10+ years']}
    df = DataFrameTransform.remove_specific_values(df, columns_to_clean)

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
    output_file_path = '/Users/fahiyeyusuf/Desktop/CLIF_data_specific_values_cleaned.csv'
    df.to_csv(output_file_path, index=False)
```
8. **Data Transformation and Skew Correction**: Use the DataFrameTransformer class to display data transformation and skew correction.
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

class Plotter:
    @staticmethod
    def visualize_distribution(data: pd.Series, column: str) -> None:
        plt.figure(figsize=(8, 6))
        sns.histplot(data, kde=True, color='skyblue')
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()

class DataFrameTransform:
    @staticmethod
    def identify_skewed_columns(df: pd.DataFrame, skew_threshold: float = 0.5, exclude_columns: list = None) -> list:
        if exclude_columns is None:
            exclude_columns = []
            
        numeric_df = df.select_dtypes(include=np.number)
        
        # Remove excluded columns from consideration
        numeric_df = numeric_df.drop(columns=exclude_columns, errors='ignore')
        
        skewness = numeric_df.skew()
        skewed_columns = skewness[abs(skewness) > skew_threshold].index.tolist()
        return skewed_columns


    @staticmethod
    def apply_best_transformation(df: pd.DataFrame, skewed_columns: list) -> pd.DataFrame:
        transformed_df = df.copy()
        for column in skewed_columns:
            if abs(transformed_df[column].skew()) > 0.5:
                transformed_df[column] = np.log1p(transformed_df[column])
        return transformed_df

if __name__ == "__main__":
    # Load DataFrame from CSV file
    file_path = '/Users/fahiyeyusuf/Desktop/CLIF_data_specific_values_cleaned.csv'
    df = pd.read_csv(file_path)

    # Example usage:
    # Step 1: Identify skewed columns, excluding certain columns
    columns_to_exclude = ['id','member_id']  # Specify columns to exclude from skewness calculation
    skewed_columns = DataFrameTransform.identify_skewed_columns(df, exclude_columns=columns_to_exclude)
    print("Skewed columns:", skewed_columns)

    # Visualize the data using Plotter
    for column in skewed_columns:
        Plotter.visualize_distribution(df[column], column)

    # Step 2: Perform transformations to reduce skew
    df_transformed = DataFrameTransform.apply_best_transformation(df, skewed_columns)

    # Visualize the transformed data
    for column in skewed_columns:
        Plotter.visualize_distribution(df_transformed[column], column)

    # Step 3: Apply transformations to reduce skew
    df = df_transformed.copy()

    # Step 4: Save a separate copy of DataFrame
    df_copy = df.copy()

    # Specify the path where you want to save the new CSV file
    output_csv_path = '/Users/fahiyeyusuf/Desktop/CLIF_data_transformed.csv'

    # Save the DataFrame to a new CSV file
    df_copy.to_csv(output_csv_path, index=False)
```
9. **Identifying and Removing Outliers**: Use the DataFrameTransform class to identify and remove outliers.
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class Plotter:
    @staticmethod
    def visualize_outliers(data: pd.Series, column: str) -> None:
        plt.figure(figsize=(8, 6))
        sns.boxplot(data[column])
        plt.title(f'Boxplot of {column}')
        plt.xlabel(column)
        plt.show()

class DataFrameTransform:
    @staticmethod
    def identify_outliers(df: pd.DataFrame, column: str) -> pd.DataFrame:
        try:
            # Convert column to numeric type and handle missing values
            df[column] = pd.to_numeric(df[column], errors='coerce')
            df = df.dropna(subset=[column])
            
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
            return outliers
        except Exception as e:
            print(f"Error processing column '{column}': {e}")
            return pd.DataFrame()

    @staticmethod
    def remove_outliers(df: pd.DataFrame, column: str) -> pd.DataFrame:
        # Convert column to numeric type and handle missing values
        df[column] = pd.to_numeric(df[column], errors='coerce')
        df = df.dropna(subset=[column])
        
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        return df

if __name__ == "__main__":
    # Load DataFrame from CSV file
    file_path = '/Users/fahiyeyusuf/Desktop/CLIF_data_transformed.csv'
    df = pd.read_csv(file_path)

    # Example usage:
    # Step 1: Visualize data to identify outliers
    # Before outlier removal
    outlier_columns = ['loan_amount', 'funded_amount', 'funded_amount_inv', 'annual_inc', 'delinq_2yrs',
                       'inq_last_6mths', 'int_rate', 'instalment', 'mths_since_last_delinq', 'open_accounts',
                       'total_accounts', 'out_prncp', 'out_prncp_inv', 'total_payment', 'total_payment_inv',
                       'total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee',
                       'collections_12_mths_ex_med', 'mths_since_last_major_derog', 'dti']  # Adjust this list based on visualization

    for column in outlier_columns:
        outliers_before = DataFrameTransform.identify_outliers(df, column)
        if not outliers_before.empty:
            print(f'Outliers in {column} before removal: {len(outliers_before)}')
            Plotter.visualize_outliers(df, column)

    # Step 2: Remove outliers
    for column in outlier_columns:
        df = DataFrameTransform.remove_outliers(df, column)

    # Step 3: Visualize data after removing outliers
    for column in outlier_columns:
        outliers_after = DataFrameTransform.identify_outliers(df, column)
        if not outliers_after.empty:
            print(f'Outliers in {column} after removal: {len(outliers_after)}')
            Plotter.visualize_outliers(df, column)
# Step 4: Save the cleaned DataFrame to a new CSV file
    cleaned_output_csv_path = '/Users/fahiyeyusuf/Desktop/CLIF_data_outlier_cleaned.csv'
    df.to_csv(cleaned_output_csv_path, index=False)
```
10. **Data Cleaning and Correlation Analysis**: Use the DataProcessor class to clean data and analyse correlation.
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class DataProcessor:
    def __init__(self, file_path: str):
        self.df = pd.read_csv(file_path)

    def preprocess_data(self, columns_to_exclude: list, correlation_threshold: float = 0.7) -> pd.DataFrame:
        # Drop non-numeric columns or convert them to numeric if possible
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        df_numeric = self.df[numeric_columns]

        # Compute the correlation matrix
        correlation_matrix = df_numeric.corr()

        # Visualize the correlation matrix
        self.visualize_correlation(correlation_matrix)

        # Identify highly correlated columns
        highly_correlated_columns = self.identify_highly_correlated(correlation_matrix, columns_to_exclude, correlation_threshold)

        # Decide which columns to remove
        columns_to_remove = list(highly_correlated_columns)
        print("Columns highly correlated and to be removed:", columns_to_remove)

        # Remove highly correlated columns from the dataset
        df_cleaned = self.df.drop(columns=columns_to_remove)

        return df_cleaned

    @staticmethod
    def visualize_correlation(correlation_matrix: pd.DataFrame) -> None:
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Matrix')
        plt.show()

    @staticmethod
    def identify_highly_correlated(correlation_matrix: pd.DataFrame, columns_to_exclude: list, correlation_threshold: float) -> set:
        highly_correlated_columns = set()
        for i in range(len(correlation_matrix.columns)):
            for j in range(i):
                if abs(correlation_matrix.iloc[i, j]) > correlation_threshold:
                    colname_i = correlation_matrix.columns[i]
                    colname_j = correlation_matrix.columns[j]
                    if colname_i not in columns_to_exclude and colname_j not in columns_to_exclude:
                        highly_correlated_columns.add(colname_i)
                        highly_correlated_columns.add(colname_j)
        return highly_correlated_columns

if __name__ == "__main__":
    # Step 1: Load the dataset and preprocess it
    file_path = '/Users/fahiyeyusuf/Desktop/CLIF_data_outlier_cleaned.csv'
    columns_to_exclude = ['id', 'loan_amount', 'total_payment', 'recoveries']
    processor = DataProcessor(file_path)
    df_cleaned = processor.preprocess_data(columns_to_exclude)
    
    # Optionally, you can save the cleaned dataset to a new CSV file
    output_csv_path = '/Users/fahiyeyusuf/Desktop/CLIF_data_cleaned.csv'
    df_cleaned.to_csv(output_csv_path, index=False)
```
### Jupyter Notebooks

1. [Data Preprocessing Notebook](Data_Preprocessing.ipynb) - Notebook containing code for data preprocessing tasks.
2. [Data Visualisation Notebook](Data_Visualisation.ipynb) - Notebook containing code for data visualisation tasks.

## File Structure
```css
exploratory-data-analysis---customer-loans-in-finance/
│   load_data.py
│   RDS_SQLAlchemy.py
│   save_loan_data.py
│   DataFrameInfo.py
│   DataFrameTransformer.py
|   Data_Null_Processing.py
|   Data_Skewness_Processing.py
|   Data_Outlier_Processing.py
|   Correlated_Data.py
|   README.md
```
## License
This project is licensed under the MIT License - see the LICENSE file for details.