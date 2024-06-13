import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv('C://dev//opencv.test//Obesity_edited.csv')
#assuming the 'NObeyesdad' column contains the target variable, and the rest are features
X = data.drop('NObeyesdad', axis=1)  #features without 'NObeyesdad'
y = data['NObeyesdad']  #target variable 'NObeyesdad'

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_classifier = RandomForestClassifier(n_estimators=400,
                                       max_depth=20,
                                       min_samples_leaf=1,
                                       min_samples_split=2,
                                       random_state=42)

rf_classifier.fit(X_train, y_train)

y_pred = rf_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("\nClassification Report:\n", classification_report(y_test, y_pred))

feature_importance = rf_classifier.feature_importances_
print("\nFeature Importance:\n", feature_importance)
