import pandas as pd
from tqdm.auto import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# Load the data
root = Path(__file__).resolve().parent.parent
file_path = f"{root / 'data' / 'UMB.csv'}"
df = pd.read_csv(file_path)

# Create a scatter plot to visualize the relationship between coumarin dose and cell viability
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='coumarinDose', y='viability', hue='cancerType', style='cancerType', s=100)
# plt.title('Scatter Plot of Cell Viability vs Coumarin Dose by Cancer Type')
plt.xlabel('Coumarin dose (μM)')
plt.ylabel('Viability (%)')
plt.legend(title='Cancer Type')
plt.grid(True)
plt.show()
