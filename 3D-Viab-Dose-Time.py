import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Load data from CSV file
data = pd.read_csv('NEWum.csv')

#data = data[data['coumarinDose'] <= 200]

# Extract the relevant columns
time_hours = data['time']
dose = data['coumarinDose']
viability_percent = data['viability']
study_id = data['ID']
author = list(data['author'])
year = list(data['year'])

d = []

for i, j in zip(author, year):
    xx = f"{i} {j}"
    if xx not in d:
        d.append(xx)

d = sorted(d)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create a colormap with enough colors
n_unique_study_ids = len(study_id.unique())
colors = plt.cm.viridis(np.linspace(0, 1, n_unique_study_ids))

# Scatter plot the data, color by study ID
for i, study_id_value in enumerate(study_id.unique()):
    mask = study_id == study_id_value
    ax.scatter(
        time_hours[mask],
        dose[mask],
        viability_percent[mask],
        c=[colors[i]],
        marker='o',
        label=f'{d[i]}'
    )

# Set labels for the axes
ax.set_xlabel('Time (hours)')
ax.set_ylabel('Coumarin dose (μM)')
ax.set_zlabel('Viability (%)')

# Add a legend
ax.legend(loc='lower right', bbox_to_anchor=(1.5, 0.35))

# Show the 3D plot
plt.show()
