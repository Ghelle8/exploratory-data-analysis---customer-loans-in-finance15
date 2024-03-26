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
