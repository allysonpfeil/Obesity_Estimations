# Obesity_Estimations
model to estimate the obesity level of individuals based on lifestyle factors

Model Engineering:
When I first initialized the RF model, the accuracy was 85%. When I assessed the variables in terms of model significance, 
 - Gender: 0.06766118
 - Age: 0.16844229
 - Family History of Obesity: 0.05051673
 - High Calorie Food: 0.02783893
 - Vegetables w/ meals: 0.14205373
 - Main Meal Quantity: 0.09677437
 - Eating Between Meals: 0.05253611
 - Smoking Status: 0.00459477
 - Water Intake: 0.09336384
 - Calorie Counting: 0.01241953
 - Physical Activity: 0.09465854
 - Hours of Technology: 0.09556981
 - Alcohol Consumption: 0.05352688
 - Transportation: 0.04004331

From these results, I decided to drop High Calorie Food and Smoking Status.


