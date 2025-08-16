"""Importing Libraries and Dataset"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler

import warnings
warnings.filterwarnings('ignore')

child = pd.read_csv('train.csv')
print(child.head())

"""Now let's check the size of the dataset"""
print(child.shape)

"""Let's check which column of the dataset contains which type of data."""
print(child.info())

"""As per the above information regarding the data in each column
   we can observe that there are no null values."""

print(child.isnull().sum())

print(child.describe().T)

print(child.info())

"""Data Cleaning"""

print(child['gender'].value_counts())

child['gender'] = child['gender'].map({'m':1,'f':2})
print(child.head())

print(child['ethnicity'].value_counts())

child['ethnicity'] = child['ethnicity'].map({'White-European':12,'?':11,'Middle Eastern':10,'Asian':9,'Black':8,'South Asian':7,'Pasifika':6,'Others':5,'Latino':4,'Hispanic':3,'Turkish':5,'others':1})
print(child.head())

child['jaundice'].value_counts()

child['jaundice'] = child['jaundice'].map({'no':0,'yes':1})
print(child.head())

print(child['austim'].value_counts())

child['austim'] = child['austim'].map({'no':0,'yes':1})
print(child.head())

print(child['contry_of_res'].value_counts())

print(child['used_app_before'].value_counts())

child['used_app_before'] = child['used_app_before'].map({'no':0,'yes':1})
print(child.head())

print(child['age_desc'].value_counts())

child['age_desc'] = child['age_desc'].map({'18 and more':100})
print(child.head())

print(child['relation'].value_counts())

child['relation'] = child['relation'].map({'Self':100,'?':80,'Parent':60,'Relative':40,'Others':20,'Health care professional':10})
print(child.head())

child.drop("contry_of_res",inplace=True, axis = 1)
print(child.head())

print(child.info())

"""Step 3:Exploratory Data Analysis"""

plt.pie(child['Class/ASD'].value_counts().values, autopct='%1.1f%%')
plt.show()
"""The pie chart shows an imbalanced dataset with approximately 80% negative and 20% positive ASD cases, 
   indicating class imbalance that may require special handling during modeling."""


import seaborn as sns

# Plot histograms for score columns
score_columns = [col for col in child.columns if 'Score' in col]
child[score_columns].hist(figsize=(15, 10))
plt.suptitle('Distribution of Scores')
plt.show()
"""The score distributions for A1 to A10 are highly skewed with most scores close to 0 or 1, 
   indicating binary-like responses with limited intermediate values."""

plt.figure(figsize=(12,8))
corr = child.corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
"""The correlation matrix shows strong positive correlations among scores, a weak positive link between age and scores, 
   and mild correlations among categorical variables. Most features are weakly or not correlated."""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set visual style
sns.set(style="whitegrid")

#  Count plot for some categorical features
categorical_features = ['gender', 'ethnicity', 'jaundice', 'austim', 'used_app_before']
for feature in categorical_features:
    plt.figure(figsize=(6,4))
    sns.countplot(x=feature, data=child)
    plt.title(f'Countplot of {feature}')
    plt.show()
"""Gender balance is nearly even, with high counts for ethnicity and prior app use, while jaundice and autism have significant disparities, 
   mostly skewed towards the '0' categories."""

#  Distribution plot for age
plt.figure(figsize=(8,4))
sns.histplot(child['age'], kde=True)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

"""The age distribution is right-skewed, concentrated among younger children, with a few older children spanning up to around 85 years,
   indicating most data points are in childhood."""

plt.subplots(figsize=(15,5))
floats = ['age', 'result']
for i, col in enumerate(floats):
  plt.subplot(1,2,i+1)
  sb.distplot(child[col])
plt.tight_layout()
plt.show()
"""Age shows a right-skewed distribution concentrated among children. The result distribution is skewed with most values between 0-15,
   indicating a majority scoring low on the measurement."""

plt.subplots(figsize=(15,5))
floats = ['age', 'result']
for i, col in enumerate(floats):
  plt.subplot(1,2,i+1)
  sb.boxplot(child[col])
plt.tight_layout()
plt.show()
"""Age displays a right-skewed distribution with most values below 50, while result shows a median around 10, 
   indicating typical score outcomes with some outliers."""

child = child[child['result']>-5]
print(child.shape)

"""Feature Engineering"""
"""Feature Engineering helps to derive some valuable features from the existing ones. 
These extra features sometimes help in increasing the performance of the model significantly and certainly help to gain deeper insights into the data."""

# This functions make groups by taking
# the age as a parameter
def convertAge(age):
    if age < 4:
        return 'Toddler'
    elif age < 12:
        return 'Kid'
    elif age < 18:
        return 'Teenager'
    elif age < 40:
        return 'Young'
    else:
        return 'Senior'

child['ageGroup'] = child['age'].apply(convertAge)

sb.countplot(x=child['ageGroup'], hue=child['Class/ASD'])
plt.show()
"""Here we can conclude that the Young and Toddler group of people have lower chances of having Autism."""


def add_feature(data):
    # Creating a column with all values zero
    data['sum_score'] = 0
    for col in data.loc[:, 'A1_Score':'A10_Score'].columns:
        # Updating the 'sum_score' value with scores
        # from A1 to A10
        data['sum_score'] += data[col]

    # Creating a random data using the below three columns
    data['ind'] = data['austim'] + data['used_app_before'] + data['jaundice']

    return data


child = add_feature(child)

sb.countplot(x=child['sum_score'], hue=child['Class/ASD'])
plt.show()
"""This histogram shows the distribution of two classes, ASD and non-ASD, across different sum scores. 
   The non-ASD group predominantly has higher counts at lower scores, while ASD counts increase notably at higher scores, especially from 9 to 10. This suggests that higher scores may be associated with ASD, indicating a potential scoring threshold for classification. 
   Overall, it highlights a clear separation between the groups based on sum scores."""

# Applying log transformations to remove the skewness of the data.
child['age'] =child['age'].apply(lambda x: np.log(x))

sb.distplot(child['age'])
plt.show()
"""This density plot depicts the age distribution, showing a approximately normal distribution centered around 3 to 3.5 years. 
   The highest density indicates most data points cluster in this age range, 
   with fewer observations at younger and older ages, 
   illustrating a typical age concentration within the sample"""


def encode_labels(data):
    for col in data.columns:

        # Here we will check if datatype
        # is object then we will encode it
        if data[col].dtype == 'object':
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])

    return data


child = encode_labels(child)

# Making a heatmap to visualize the correlation matrix
plt.figure(figsize=(10, 10))
sb.heatmap(child.corr() > 0.8, annot=True, cbar=False)
plt.show()
"""This heatmap displays correlations among variables, 
   with high values on the diagonal indicating perfect self-correlation and varying degrees of positive and negative relationships among features. 
   Key correlations include between scores, age, and demographic factors, suggesting interconnected influences in the dataset."""

removal = ['ID', 'age_desc', 'used_app_before', 'austim']
features = child.drop(removal + ['Class/ASD'], axis=1)
target = child['Class/ASD']

X_train, X_val, Y_train, Y_val = train_test_split(features, target, test_size = 0.2, random_state=10)

# As the data was highly imbalanced we will balance it by adding repetitive rows of minority class.
ros = RandomOverSampler(sampling_strategy='minority',random_state=0)
X, Y = ros.fit_resample(X_train,Y_train)
print(X.shape, Y.shape)

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_val = scaler.transform(X_val)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

models = [
    LogisticRegression(),
    RandomForestClassifier(),
    SVC(probability=True)
]
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

for model in models:
    model.fit(X_imputed, Y)
    print(f'{model} : ')
    print('Training ROC-AUC Score : ', metrics.roc_auc_score(Y, model.predict(X_imputed)))
    print('Validation ROC-AUC Score : ', metrics.roc_auc_score(Y_val, model.predict(imputer.transform(X_val))))
    print()

from sklearn.impute import SimpleImputer

# Assuming X_val is your validation feature set
imputer = SimpleImputer(strategy='mean')
X_val_imputed = imputer.fit_transform(X_val)

from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

ConfusionMatrixDisplay.from_estimator(models[0], X_val_imputed, Y_val)
plt.show()

