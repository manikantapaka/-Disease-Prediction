import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load datasets
train_df = pd.read_csv("/content/training_data.csv")
test_df = pd.read_csv("/content/test_data.csv")

# Clean unnamed index columns if present
train_df = train_df.loc[:, ~train_df.columns.str.contains('^Unnamed')]
test_df = test_df.loc[:, ~test_df.columns.str.contains('^Unnamed')]

# Split features and labels
X_train = train_df.iloc[:, :-1]
y_train = train_df.iloc[:, -1]
X_test = test_df.iloc[:, :-1]
y_test = test_df.iloc[:, -1]

# Hybrid model using VotingClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
model = VotingClassifier(estimators=[('rf', rf), ('gb', gb)], voting='hard')

# Train model
model.fit(X_train, y_train)

# Evaluate on test data
y_pred = model.predict(X_test)
print("✅ Test Accuracy:", accuracy_score(y_test, y_pred))
print("\n✅ Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n✅ Classification Report:\n", classification_report(y_test, y_pred))

# Plot feature correlation heatmap
sn.heatmap(train_df.iloc[:, :-1].corr(), cmap="YlGnBu")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()
