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
    file_path = '/Users/fahiyeyusuf/Desktop/CLIF_data.csv'
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
