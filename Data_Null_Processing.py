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
