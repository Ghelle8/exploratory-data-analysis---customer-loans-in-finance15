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
