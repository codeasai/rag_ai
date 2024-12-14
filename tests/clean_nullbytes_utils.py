# Before running this code, PLEASE ensure you have a backup of the original file "test_utils.py"

with open('test_utils.py', 'r') as original_file:
    data = original_file.read()

cleaned_data = data.replace('\x00', '')  # Removing all null bytes

with open('cleaned_test_utils.py', 'w') as cleaned_file:
    cleaned_file.write(cleaned_data)