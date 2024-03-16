import pandas as pd

class DataTransform:
    @staticmethod
    def convert_to_numeric(df, columns):
        for col in columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    
    @staticmethod
    def convert_to_datetime(df, columns):
        for col in columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        return df
    
    @staticmethod
    def convert_to_categorical(df, columns):
        for col in columns:
            df[col] = df[col].astype('category')
        return df
    
    @staticmethod
    def remove_excess_symbols(df, columns, symbols):
        for col in columns:
            df[col] = df[col].str.replace(symbols, '')
        return df

# Example usage:
# Load data
df = pd.read_csv('data.csv')

# Print column names to verify
print(df.columns)

# Instantiate DataTransform class
transformer = DataTransform()

# Define columns needing format adjustments
numeric_columns = ['loan_amount', 'funded_amount', 'funded_amount_inv', 'annual_inc', 'last_payment_amount', 'open_accounts', 'total_accounts', 'out_prncp', 'out_prncp_inv', 'total_payment', 'total_payment_inv', 'total_rec_prncp', 'total_rec_int' ]  # Update with actual column names
date_columns = ['issue_date', 'last_payment_date', 'next_payment_date', 'last_credit_pull_date' ]  # Update with actual column names
categorical_columns = ['grade', 'sub_grade', 'home_ownership', 'verification_status', 'loan_status', 'application_type', 'payment_plan', 'grade', 'sub_grade' ]  # Update with actual column names
columns_with_symbols = ['purpose']  # Update with actual column names

# Apply transformations
df = transformer.convert_to_numeric(df, numeric_columns)
df = transformer.convert_to_datetime(df, date_columns)
df = transformer.convert_to_categorical(df, categorical_columns)
df = transformer.remove_excess_symbols(df, columns_with_symbols, '@#$')

# Verify transformations
print(df.dtypes)  # Check data types of columns
print(df.head())  # View sample of transformed data
