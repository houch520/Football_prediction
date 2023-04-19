import csv
import subprocess
import pandas as pd
import os
import shutil
from datetime import datetime
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

# Define the input string
input_str = 'J2'
input_nth = ''
input_r = 'R'

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
