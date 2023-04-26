import csv
import subprocess
import pandas as pd
import os
import shutil
from datetime import datetime

# Define the input string
input_str = 'E2'
input_nth = ''
input_r = 'R'

#Please make sure your Source file is sorted by time
def SortFile(filename):
    # Read the CSV file
    df = pd.read_csv('Source\\'+filename+'.csv')

    backup_filename = filename+'_backup.csv'
    os.rename('Source\\'+filename+'.csv', 'Source\\'+backup_filename)

    # Convert the "Date" column to datetime format
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    # Sort the DataFrame by the "Date" column in ascending order
    df = df.sort_values('Date', ascending=True)

    # Save the sorted data to a CSV file
    df.to_csv('Source\\'+filename+'.csv', index=False,date_format='%d/%m/%Y')

    # Move a file to another folder
    original_file_path = 'Source\\'+backup_filename
    new_folder_path = 'Source\\backup'
    shutil.move(original_file_path, new_folder_path)


#Don use this function if possible
#SortFile(input_str)

# Read the CSV file
df = pd.read_csv('Source\\'+input_str+input_nth+'.csv')

# Reverse the order of the rows
df = df.iloc[::-1]

# Save the data to a new CSV file
df.to_csv('Source\\'+input_str+input_r+'.csv', index=False)

# Run the command and pass the input string as input
subprocess.call(['python', 'Predict_ver1.py', input_str, input_nth])
# Run the command and pass the input string as input
subprocess.call(['python', 'Predict_ver1.py', input_str, input_r])

# Read CSV files
csv_file1 = 'Result\predictions'+input_str+'.csv'
csv_file2 = 'Result\predictions'+input_str+input_r+'.csv'

# Loop through all rows in the two CSV files and compare the percentage of H
with open(csv_file1, 'r') as f1, open(csv_file2, 'r') as f2:
    reader1 = csv.DictReader(f1)
    reader2 = csv.DictReader(f2)
    for row1, row2 in zip(reader1, reader2):
        if row1['PredictResult'] == row1['Home']:
            file1_percentage = float(row1['H'].strip('%'))
            file2_percentage = float(row2['H'].strip('%'))
        elif row1['PredictResult'] == row1['Away']:
            file1_percentage = float(row1['A'].strip('%'))
            file2_percentage = float(row2['A'].strip('%'))
        else:
            file1_percentage = float(row1['D'].strip('%'))
            file2_percentage = float(row2['D'].strip('%'))
        
        # Compare the percentage of H for the two rows
        if file1_percentage > file2_percentage:
            print('{},Good,{}-{}'.format(row1['PredictResult'],file1_percentage,file2_percentage))
        elif file1_percentage < file2_percentage:
            print('{},Bad,{}-{}'.format(row1['PredictResult'],file1_percentage,file2_percentage))
        else:
            print('{},Draw,{}-{}'.format(row1['PredictResult'],file1_percentage,file2_percentage))