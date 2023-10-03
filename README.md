# Obesity_Estimations
model to estimate the obesity level of individuals based on lifestyle factors

This model uses data from the UCI machine learning repository to classify individual obesity based on lifestyle factors. The original dataset can be found at:
https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition

The version found there contains the variables 'weight' and 'height'; however, I removed them since they are too indicative of BMI class, which is the target variable. FWIW, my model was over 95% accurate when including the weight and height data. 
All of the data can be found in the 'obesity_edited.csv' file in the repository.

Model Engineering:
The accuracy of this model is 86%. The following is the breakdown of the features and their importance:
 - Gender: 0.06839598
 - Age: 0.16534347
 - Family History of Obesity: 0.05003454
 - High Calorie Food: 0.02681835
 - Vegetables w/ meals: 0.14311227
 - Main Meal Quantity: 0.09663267
 - Eating Between Meals: 0.05286194
 - Smoking Status: 0.00449358
 - Water Intake: 0.09418564
 - Calorie Counting: 0.01127133
 - Physical Activity: 0.09667078
 - Hours of Technology: 0.09612716
 - Alcohol Consumption: 0.05318949
 - Transportation: 0.0408628



