# Exploratory Data Analysis - Customer Loans in Finance

## Table of Contents

1. [Description](#description)
2. [Installation](#installation)
3. [Usage](#usage)
4. [File Structure](#file-structure)
5. [License](#license)

## Description

This project aims to perform exploratory data analysis on customer loans in the finance domain. It includes functions to load data from a CSV file into a Pandas DataFrame and extract data from an RDS database, as well as save data to a CSV file. Through this project, I've learned how to work with data from financial databases, connect to databases using SQLAlchemy, and perform data analysis using Pandas.

## Installation

To install the required dependencies, run:

pip install pandas sqlalchemy


## Usage

1. **Loading Data**: Use the `load_data` function to load data from a CSV file into a Pandas DataFrame.

```python
import pandas as pd

def load_data(file_path):
    # Implementation here...

# Example usage:
file_path = 'data.csv'
data_df = load_data(file_path)
```
2. **Connecting to RDS Database**: Create an instance of RDSDatabaseConnector with your RDS credentials and use the connect method to establish a connection.

```python
from sqlalchemy import create_engine

class RDSDatabaseConnector:
    # Implementation here...

# Example usage:
credentials = {
    'RDS_HOST': 'eda-projects.cq2e8zno855e.eu-west-1.rds.amazonaws.com',
    'RDS_PASSWORD': 'EDAloananalyst',
    'RDS_USER': 'loansanalyst',
    'RDS_DATABASE': 'payments',
    'RDS_PORT': 5432
}

connector = RDSDatabaseConnector(credentials)
connector.connect()
```
3. **Extracting Data from RDS Database**: Use the extract_data method to fetch data from the RDS database.
```python
data_df = connector.extract_data()
```
4. **Saving Data to CSV**: Use the save_to_csv function to save a DataFrame to a CSV file
```python
def save_to_csv(dataframe, file_path):
    # Implementation here...

# Example usage:
file_path = 'data.csv'
save_to_csv(data_df, file_path)
```
## File Structure
```css
exploratory-data-analysis---customer-loans-in-finance/
│   load_data.py
│   RDS_SQLAlchemy.py
│   save_loan_data.py
│   README.md
│   ...
```
## License
This project is licensed under the MIT License - see the LICENSE file for details.

```sql
3. Save the changes to your README file.
4. Add and commit the changes using Git:

```bash
git add README.md
git commit -m "Update README with project information"
git push origin main
```

