import pandas as pd
from tqdm.auto import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
file_path = 'NEWau.csv'
df = pd.read_csv(file_path)

# Select the columns of interest
selected_columns = df[['coumarinDose', 'time', 'viability',]]

# Calculate the correlation matrix
correlation_matrix = selected_columns.corr()

# Create a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
plt.title('Heatmap of Correlations')
plt.show()
