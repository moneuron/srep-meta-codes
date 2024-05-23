import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = 'all.csv'
df = pd.read_csv(file_path)

# Set the color palette to a publication-friendly palette
sns.set_palette('colorblind')

# Create FacetGrid with scatterplots and regression lines
g = sns.FacetGrid(df, col='coumarin', hue='cellLine', palette='colorblind', height=6, aspect=1.5)
g.map(sns.scatterplot, 'coumarinDose', 'viability')
g.map(sns.regplot, 'coumarinDose', 'viability', scatter=False)  # Add regression line without scatter points

# Set axis labels and titles
g.set_axis_labels('Coumarin dose (μM)', 'Viability (%)')
g.set_titles(col_template='')

# Set the same labels for both x and y axes
for ax in g.axes.flat:
    ax.set_xlabel('Coumarin dose (μM)')
    ax.set_ylabel('Viability (%)')

plt.tight_layout()
plt.show()
