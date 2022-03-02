# Insurance-Claims-Severity
The objective of this project is to use supervised learning techniques to predict how severe is a claim in "loss", given some information about a particular claim.  Efficiencies in insurance claims severity analysis can help provide suitable insurance packages to customers and provide targeted assistance to better serve them. Finally, we would like to build a model that can predict severity of claims so as to improve the claims service to ensure a worry-free customer experience.

# Dataset
The data set to be explored is from Kaggle in https://www.kaggle.com/c/allstate-claims-severity/data; The training dataset consists of 130 attributes (features) and the loss value for each observation. The dataset contains 188,318 observations where each row represents an insurance claim. This means each claim is a process that requires 130 different information. The dataset has been anonymized.

# Data Exploration and Feature Engineering
1. encode all the categorical features into numerical values
2. check for missing values, duplicate values, irrelevant values, outliers
3. The class label is imbalanced shown as below, the model is likely to be trained on a much larger number of low cost claims, and will be less likely to successfully predict the price for the most expensive claims
![image](https://user-images.githubusercontent.com/52012182/156225513-c1c565db-d6a1-4b9c-af2a-836b57597558.png)
![image](https://user-images.githubusercontent.com/52012182/156225552-bb705ac4-8c38-47a8-87f8-22f48d170323.png)

   In this case log transformation was applied to make the class label normal, shown as below:

![image](https://user-images.githubusercontent.com/52012182/156225742-3231e152-1ce3-4be8-9a2b-f5583bd03f5b.png)

4. Perfom correlation analysis on continuous features to find the top correlated parirs. Then remove highly correlated continuous features which also have lower variance.
       ![image](https://user-images.githubusercontent.com/52012182/156226914-d4967caf-ea3c-4d34-891b-199c4037c9e1.png)   ![image](https://user-images.githubusercontent.com/52012182/156227094-faa3f67f-30af-4490-aec5-831a61962217.png)

5. Feature transoformation and scaling on contunuous features: Apply box-cox transformations on the skewed features to transform them into normal distrubuted, then apply standard scaling.
6. Perfom correlation analysis on categorical features to find the top correlated parirs. Then remove highly correlated categorical features which also have lower variance.
7. Check feature importance, then remove feaures with low importance
   ![image](https://user-images.githubusercontent.com/52012182/156227906-4e583afb-59ca-4f6a-b99d-771844d4384e.png)

# Modeling
- There are 3 machine learning models used to predict the loss: Linear Regression, Random Forest, XGBoost
- Use 5-fold Cross-Validation to evaluate the model performance

# Evalueation
- The models in this project use the mean absolute error (MAE) between the predicted loss and the actual loss for each claim in the test set. The goal was to minimize the MAE in our model’s predictions.
- The XGBoost have better performace in MAE compared to Random Forest and Linear Regression
