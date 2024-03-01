import pandas as pd

def load_data(file_path):
    """
    Load data from a CSV file into a Pandas DataFrame.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pandas.DataFrame: DataFrame containing the loaded data.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print("Error loading data:", e)
        return None

# Example usage:
file_path = 'data.csv'  # Provide the path to your CSV file
data_df = load_data(file_path)

# Print the shape of the DataFrame
print("Shape of the DataFrame:", data_df.shape)

# Print a sample of the data
print("Sample of the data:")
print(data_df.head())
