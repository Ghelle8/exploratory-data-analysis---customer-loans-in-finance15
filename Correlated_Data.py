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
