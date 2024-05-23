from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
from tqdm.auto import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load the data
file_path = 'all.csv'
data = pd.read_csv(file_path)
# Encode categorical variables
label_encoder = LabelEncoder()
data['cancerType'] = label_encoder.fit_transform(data['cancerType'])
data['coumarin'] = label_encoder.fit_transform(data['coumarin'])

# Define 'highly responsive' (1) and 'lowly responsive' (0) based on viability
# Assuming that lower viability means higher responsiveness
# This threshold can be adjusted
viability_threshold = data['viability'].median()
data['response'] = (data['viability'] < viability_threshold).astype(int)

# Select features and target
X = data[['time', 'coumarinDose', 'cancerType', 'coumarin']]
y = data['response']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42)

# Train the model
rf_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print('Accuracy of the model:', accuracy)
print('Classification Report:\n', class_report)

feature_importances = rf_classifier.feature_importances_
feature_names = X.columns

top_n_features = 8

# Sort feature importances and feature names in descending order
sorted_indices = np.argsort(feature_importances)[::-1]
sorted_feature_importances = feature_importances[sorted_indices]
sorted_feature_names = feature_names[sorted_indices]

plt.figure(figsize=(8, 8))
sns.barplot(x=sorted_feature_importances[:top_n_features], y=sorted_feature_names[:top_n_features], palette='viridis')

# Highlight top features with a different color
import seaborn as sns

# Generate a darker shade of grey color palette
n_colors = 5  # Adjust the number of colors as needed
base_color = sns.color_palette('Greys', n_colors=1)  # Base color from 'Greys' palette
dark_greys = sns.dark_palette(base_color[0], n_colors=n_colors)

palette = dark_greys
bars = plt.barh(sorted_feature_names[:top_n_features], sorted_feature_importances[:top_n_features], color=palette)

# Add annotations to the bars
for bar, importance in zip(bars, sorted_feature_importances[:top_n_features]):
    plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{importance:.3f}', ha='left', va='center', fontsize=14)

plt.xlabel('Feature Importance')
plt.title('Top Random Forest Feature Importance')
plt.grid(axis='x', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

from sklearn.metrics import classification_report
class_report_dict = classification_report(y_test, y_pred, output_dict=True)
class_report_df = pd.DataFrame(class_report_dict).transpose()

# Plotting the heatmap for precision, recall, and F1-score
plt.figure(figsize=(8, 6))
sns.heatmap(class_report_df.iloc[:-1, :-1], annot=True, cmap='coolwarm', cbar=True, fmt='.2f')
plt.title('Classification Report: Precision, Recall, and F1-score')
plt.xlabel('Metrics')
plt.ylabel('Classes')
plt.show()
