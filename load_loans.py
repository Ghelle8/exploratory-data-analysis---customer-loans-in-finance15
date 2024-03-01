import pandas as pd

def load_csv_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file into a Pandas DataFrame.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the loaded data.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"Error loading data: {e}")
    return None

if __name__ == "__main__":
    # Example usage:
    file_path = 'data.csv'  # Provide the path to your CSV file
    data_df = load_csv_data(file_path)

    if data_df is not None:
        # Print the shape of the DataFrame
        print("Shape of the DataFrame:", data_df.shape)

        # Print a sample of the data
        print("Sample of the data:")
        print(data_df.head())
