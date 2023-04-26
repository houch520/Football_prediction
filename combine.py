import pandas as pd

# Load the Excel file into a pandas DataFrame
excel_file = pd.ExcelFile('All.xlsx')

# Get the list of sheet names in the Excel file
sheet_names = excel_file.sheet_names

# Create an empty list to hold the DataFrames
df_list = []

# Loop through each sheet name and read the data into a DataFrame
for sheet_name in sheet_names:
    df = excel_file.parse(sheet_name)
    df_list.append(df)

# Concatenate all DataFrames into a single DataFrame
combined_df = pd.concat(df_list, ignore_index=True)

# Export the combined data to a CSV file
combined_df.to_csv('combined_data.csv', index=False)