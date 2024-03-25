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
