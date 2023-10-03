#RF model to predict weight classification from lifestyle factors#
#By: Allyson Pfeil

#import packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

#load data from a CSV into a pandas DataFrame
data = pd.read_csv('C://dev//opencv.test//Obesity_edited.csv')

#assuming the 'NObeyesdad' column contains the target variable, and the rest are features
X = data.drop('NObeyesdad', axis=1)  #features without 'NObeyesdad'
y = data['NObeyesdad']  #target variable 'NObeyesdad'

#split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#create a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=400,
                                       max_depth=20,
                                       min_samples_leaf=1,
                                       min_samples_split=2,
                                       random_state=42)

#train the classifier on the training data
rf_classifier.fit(X_train, y_train)

#make predictions on the test data
y_pred = rf_classifier.predict(X_test)

#evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

#print a classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

#assess feature importance
feature_importance = rf_classifier.feature_importances_
print("\nFeature Importance:\n", feature_importance)

###end###
