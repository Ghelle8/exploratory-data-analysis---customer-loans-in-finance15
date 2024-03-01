import pandas as pd
from sqlalchemy import create_engine

def save_to_csv(dataframe: pd.DataFrame, file_path: str) -> None:
    """
    Save DataFrame to a CSV file.

    Args:
        dataframe (pandas.DataFrame): DataFrame to be saved.
        file_path (str): Path to save the CSV file.
    """
    dataframe.to_csv(file_path, index=False)

class RDSDatabaseConnector:
    def __init__(self, credentials: dict) -> None:
        self.host = credentials['RDS_HOST']
        self.port = credentials['RDS_PORT']
        self.database = credentials['RDS_DATABASE']
        self.user = credentials['RDS_USER']
        self.password = credentials['RDS_PASSWORD']
        self.engine = None
    
    def connect(self) -> None:
        """
        Establishes a connection to the RDS database.
        """
        try:
            connection_string = f'postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}'
            self.engine = create_engine(connection_string)
            print("Successfully connected to the database!")
        except Exception as e:
            print("Error connecting to the database:", e)
    
    def extract_data(self) -> pd.DataFrame:
        """
        Extracts data from the RDS database and returns it as a DataFrame.
        """
        try:
            if self.engine is None:
                print("Database connection not established. Please connect first.")
                return None
            
            # Query to fetch data from loan_payments table
            query = "SELECT * FROM loan_payments"
            
            # Execute the query and fetch data into a DataFrame
            df = pd.read_sql(query, self.engine)
            
            return df
        except Exception as e:
            print("Error extracting data from the database:", e)
            return None

    def close_connection(self) -> None:
        """
        Closes the connection to the RDS database.
        """
        if self.engine is not None:
            self.engine.dispose()
            print("Connection to the database closed.")

# Example usage:
# Assuming credentials is a dictionary containing the RDS credentials
credentials = {
    'RDS_HOST': 'eda-projects.cq2e8zno855e.eu-west-1.rds.amazonaws.com',
    'RDS_PASSWORD': 'EDAloananalyst',
    'RDS_USER': 'loansanalyst',
    'RDS_DATABASE': 'payments',
    'RDS_PORT': 5432
}

# Create an instance of RDSDatabaseConnector
connector = RDSDatabaseConnector(credentials)

# Connect to the database
connector.connect()

# Extract data from the loan_payments table
data_df = connector.extract_data()

# Call the save_to_csv() function
file_path = 'data.csv'  # Provide the desired file path
save_to_csv(data_df, file_path)

