########################################### IMPORTING LIBRARIES ###########################################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

######################################## EXPLORATORY DATA ANALYSIS ########################################

data = pd.read_csv('C:/Users/ASUS/Downloads/DC++ Downloads/Projects/Diabetes Predictor/diabetes.csv')
data.columns
data.head()
print(data.shape)

# Number of people tested positive and negative for diabetes
data.groupby('Outcome').size()
sns.countplot(data['Outcome'], label="Count")

# Check for null values
data.isnull().sum()

############################################## VISUALIZATION ##############################################

colormap = plt.cm.viridis
plt.figure(figsize=(15,15))
plt.title("Corrrelation map")
sns.heatmap(data.corr(), cmap=colormap, annot=True, linewidths=0.1, vmax=1.0, 
            square=True, linecolor='black')
data.hist(figsize=(9, 9))

################################################# CLEANING ################################################

"""
We have to check how many people with 0 value in a particular attribute
fall under 0 and 1 value of the outcome variable
"""

# Blood Pressure of a person can't be zero, so we first check for that.
print("Total :", data[data.BloodPressure == 0].shape[0])
print(data[data.BloodPressure == 0].groupby('Outcome')['Age'].count())

# Even Glucose levels of a person can't be zero.
print("Total : ", data[data.Glucose == 0].shape[0])
print(data[data.Glucose == 0].groupby('Outcome')['Age'].count())

# Neither can skin fold thickness of a person.
print("Total : ", data[data.SkinThickness == 0].shape[0])
print(data[data.SkinThickness == 0].groupby('Outcome')['Age'].count())

# BMI of a person also can not be zero unless he/she is severly underweight.
print("Total : ", data[data.BMI == 0].shape[0])
print(data[data.BMI == 0].groupby('Outcome')['Age'].count())

# Insuling count of a person could be 0 on rare occassions.
print("Total : ", data[data.Insulin == 0].shape[0])
print(data[data.Insulin == 0].groupby('Outcome')['Age'].count())

# Removing Outliers of zero BloodPressure, BMI and Glucose
diabetes_mod = data[(data.BloodPressure != 0) & (data.BMI != 0) & (data.Glucose != 0)]
print(diabetes_mod.shape)

################################################# FEATURES ################################################

feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
				'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
X = diabetes_mod[feature_names]
y = diabetes_mod.Outcome

################################################# SCALING #################################################

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

########################################### SVM MODEL & METRICS ###########################################

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

############################################## FITTING MODEL ##############################################

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = diabetes_mod.Outcome, random_state=0)

# Using rbf kernel by default

# Setting parameter C = 1000
model = SVC(C = 1000, gamma = 'scale')
model.fit(X_train, y_train)
print("Accuracy on training set: {: .2f}".format(model.score(X_train, y_train))) #1.00
print("Accuracy on test set: {: .2f}".format(model.score(X_test, y_test)))       #0.71
# Evidence of overfitting as it fits training data significantly better than test

# Setting parameter C = 500
model = SVC(C = 500, gamma = 'scale')
model.fit(X_train, y_train)
print("Accuracy on training set: {: .2f}".format(model.score(X_train, y_train))) #0.99
print("Accuracy on test set: {: .2f}".format(model.score(X_test, y_test)))       #0.73
# Again overfitting

# Setting parameter C = 100
model = SVC(C = 100, gamma = 'scale')
model.fit(X_train, y_train)
print("Accuracy on training set: {: .2f}".format(model.score(X_train, y_train))) #0.98
print("Accuracy on test set: {: .2f}".format(model.score(X_test, y_test)))       #0.75
# Still overfitting

# Setting parameter C = 5
model = SVC(C = 5, gamma = 'scale')
model.fit(X_train, y_train)
print("Accuracy on training set: {: .2f}".format(model.score(X_train, y_train))) #0.86
print("Accuracy on test set: {: .2f}".format(model.score(X_test, y_test)))       #0.78
# Optimal values

y_pred = model.predict(X_test)

################################################# ACCURACY ################################################

print(accuracy_score(y_test, y_pred))                                            #0.78
kfold = KFold(n_splits = 10, random_state = 0)
print(cross_val_score(model, X, y, cv = kfold, scoring = 'accuracy').mean())     #0.75
