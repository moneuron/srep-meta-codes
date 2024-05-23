import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
file_path = 'NEWum.csv'
df = pd.read_csv(file_path)

cancer_type_counts = df['cancerType'].value_counts()

# Create a light blue spectrum using numpy's linspace
light_blue_spectrum = plt.cm.Greens(np.linspace(0.6, 0, len(cancer_type_counts)))

# Set a better font and font size
plt.rcParams['font.size'] = 12

# Create a pie chart with improved text positioning, black font color, and darker outline
plt.figure(figsize=(10, 8))
plt.pie(cancer_type_counts, labels=None, autopct='%1.1f%%', startangle=100, colors=light_blue_spectrum, textprops={'color': 'black', 'fontsize': 8}, wedgeprops={'edgecolor': 'black'})

# Add labels with better positioning
plt.title('Distribution of Cancer Types', pad=20)
plt.legend(cancer_type_counts.index, loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()
