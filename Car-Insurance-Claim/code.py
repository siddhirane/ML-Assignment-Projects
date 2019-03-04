# --------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Code starts here
df = pd.read_csv(path)
print(df.info())
#df['INCOME'] = df['INCOME'].str.lstrip('$')
#df['HOME_VAL'] = df['HOME_VAL'].str.lstrip('$')
#df['BLUEBOOK'] = df['BLUEBOOK'].str.lstrip('$')
#df['OLDCLAIM'] = df['OLDCLAIM'].str.lstrip('$')
#df['CLM_AMT'] = df['CLM_AMT'].str.lstrip('$')
cols = ['INCOME', 'HOME_VAL', 'BLUEBOOK', 'OLDCLAIM', 'CLM_AMT']
df[cols] = df[cols].replace({'\$': '', ',': ''}, regex=True)
print(df.head(10))
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
count = df['CLAIM_FLAG'].value_counts()
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.3, random_state=6)

# Code ends here


# --------------
# Code starts here
cols = ['INCOME', 'HOME_VAL', 'BLUEBOOK', 'OLDCLAIM', 'CLM_AMT']

for i in cols:
    X_train[i]=X_train[i].astype('float')
    X_test[i]=X_test[i].astype('float')

print(X_train.isnull().sum())
print(X_test.isnull().sum())

# Code ends here


# --------------
# Code starts here
X_train.dropna(subset=['YOJ','OCCUPATION'], inplace=True)
X_test.dropna(subset=['YOJ','OCCUPATION'], inplace=True)
y_train = y_train[X_train.index]
y_test = y_test[X_test.index]
cols = ['AGE', 'CAR_AGE', 'INCOME', 'HOME_VAL']
for i in cols:
    X_train[i].fillna((X_train[i].mean()), inplace = True)
    X_test[i].fillna((X_test[i].mean()), inplace = True)


# Code ends here


# --------------
from sklearn.preprocessing import LabelEncoder
columns = ["PARENT1","MSTATUS","GENDER","EDUCATION","OCCUPATION","CAR_USE","CAR_TYPE","RED_CAR","REVOKED"]

# Code starts here
le = LabelEncoder()
for i in columns:
    X_train[i] = le.fit_transform(X_train[i].astype('str'))
    X_test[i] = le.transform(X_test[i].astype('str'))

# Code ends here



# --------------
from sklearn.metrics import precision_score 
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression



# code starts here 
model = LogisticRegression(random_state = 6)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
score = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)


# Code ends here


# --------------
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# code starts here
smote = SMOTE(random_state = 6)
X_train, y_train = smote.fit_sample(X_train, y_train)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit(X_test)


# Code ends here


# --------------
# Code Starts here
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
score = accuracy_score(y_test, y_pred)

# Code ends here


