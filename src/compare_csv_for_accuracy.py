import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

'''
This is a script used to generate the MAPE between manually labelled data and Cilia.io's labeled data.
'''

# Load the CSVs (adjust filenames and index_col as needed)
df1 = pd.read_csv('MANUAL_DATA_FILEPATH.csv', index_col="cilia_id")
df2 = pd.read_csv("CILIA_IO_DATA_FILEPATH.csv", index_col="cilia_id")

# Merge on cilia_id (assuming both have a column named 'cilia_id')
merged = pd.merge(df1.reset_index(), df2.reset_index(), on="cilia_id", suffixes=("_1", "_2"))

print(merged)

# Extract frequency and mean_cilia_length columns
freq1 = merged["frequency_1"]
freq2 = merged["frequency_2"]

print(freq1)
print(freq2)

mean_len1 = merged["mean_cilia_length_1"]
mean_len2 = merged["mean_cilia_length_2"]

# Compute MSE and MAPE
freq_mse = mean_squared_error(freq1, freq2)
freq_mape = mean_absolute_percentage_error(freq1, freq2) * 100

len_mse = mean_squared_error(mean_len1, mean_len2)
len_mape = mean_absolute_percentage_error(mean_len1, mean_len2) * 100

print(f"Frequency MSE: {freq_mse:.4f}, MAPE: {freq_mape:.2f}%")
print(f"Mean Cilia Length MSE: {len_mse:.4f}, MAPE: {len_mape:.2f}%")
