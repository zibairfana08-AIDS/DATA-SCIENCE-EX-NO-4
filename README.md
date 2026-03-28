# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.

STEP 2:Clean the Data Set using Data Cleaning Process.

STEP 3:Apply Feature Scaling for the feature in the data set.

STEP 4:Apply Feature Selection for the feature in the data set.

STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1

2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.

3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.

4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.

The feature selection techniques used are:

1.Filter Method

2.Wrapper Method

3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("bmi.csv")
df.head()
```
<img width="265" height="203" alt="image" src="https://github.com/user-attachments/assets/23ecaff7-0ab2-4e9b-832b-ed90876abbf6" />

```
df.dropna()
```
<img width="314" height="412" alt="image" src="https://github.com/user-attachments/assets/23984e0a-9450-4e3d-82c5-2f2b6a88eeb2" />

```
max_vals=np.max(np.abs(df[['Height','Weight']]))
max_vals
```
<img width="62" height="26" alt="image" src="https://github.com/user-attachments/assets/51a51f09-4889-4750-8564-bab31dd9b16b" />

```
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)
```
<img width="286" height="341" alt="image" src="https://github.com/user-attachments/assets/73e045f9-d32d-4196-8c6d-6d831e265e24" />

```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
<img width="279" height="334" alt="image" src="https://github.com/user-attachments/assets/d258211f-f630-4fa1-8d2e-dcd15800f466" />

```
from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```
<img width="299" height="415" alt="image" src="https://github.com/user-attachments/assets/f38a8ef6-55a7-496b-a591-275ee3ce28f5" />

```
df3=pd.read_csv("bmi.csv")
from sklearn.preprocessing import MaxAbsScaler
scaler=MaxAbsScaler()
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
df3
```
<img width="285" height="413" alt="image" src="https://github.com/user-attachments/assets/dc137cc8-e878-404d-95d5-ca0f0a0861d9" />

```
df4=pd.read_csv("bmi.csv")
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df4[['Height','Weight']]=scaler.fit_transform(df4[['Height','Weight']])
df4
```

<img width="302" height="409" alt="image" src="https://github.com/user-attachments/assets/d346bb51-cbd6-4b98-ab01-2871abfbbb4d" />

```
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV,LassoCV,Ridge,Lasso
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import chi2
import pandas as pd
df=pd.read_csv("titanic_dataset.csv")
df.columns
```

<img width="544" height="60" alt="image" src="https://github.com/user-attachments/assets/34142b68-f991-4fbc-a362-a5c973c1d7a7" />

```
df.shape
```

<img width="99" height="26" alt="image" src="https://github.com/user-attachments/assets/b79c56aa-f5b1-40f0-b0c5-2afc864ade74" />

```
x=df.drop("Survived",axis=1)
y=df['Survived']
df1=df.drop(["Name","Sex","Ticket","Cabin","Embarked"],axis=1)
df1.columns
```

<img width="686" height="31" alt="image" src="https://github.com/user-attachments/assets/2d8ace8b-58cb-472e-96de-97958dac42a8" />

```
df1['Age'].isnull().sum()
```

<img width="112" height="31" alt="image" src="https://github.com/user-attachments/assets/6a47a124-ecba-49c3-8b6e-71aec3ddfd3f" />

```
df1['Age'].fillna(method='ffill')
```

<img width="277" height="212" alt="image" src="https://github.com/user-attachments/assets/d41917eb-dc4b-43f1-b51a-587d7e99bb32" />

```
df1['Age']=df1['Age'].fillna(method='ffill')
df1['Age'].isnull().sum()
```

<img width="114" height="37" alt="image" src="https://github.com/user-attachments/assets/c94fdbfc-e67e-4dcb-af1a-6a8351939845" />

```
df1.columns
```

<img width="681" height="34" alt="image" src="https://github.com/user-attachments/assets/56078918-0ee3-43a0-80a5-b6ad98ac70d1" />

```
cols=df1.columns.tolist()
cols[-1],cols[1]=cols[1],cols[-1]
df1=df1[cols]
df1.columns
```

<img width="691" height="37" alt="image" src="https://github.com/user-attachments/assets/6f90bd6b-0811-4798-9608-06a42c8ce862" />

```
x=df1.iloc[:,0:6]
y=df1.iloc[:,6]
x.columns
```

<img width="617" height="26" alt="image" src="https://github.com/user-attachments/assets/f020105d-38dd-4af4-8fd0-c98cba89bffc" />

```
y=y.to_frame()
y.columns
```

<img width="275" height="33" alt="image" src="https://github.com/user-attachments/assets/e5fff4ff-90a2-40f5-8b1a-cf9e638ff6f2" />

```
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
data=pd.read_csv("titanic_dataset.csv")
data = data.dropna()
x=data.drop(['Survived','Name','Ticket'],axis=1)
y=data['Survived']
x
```

<img width="591" height="401" alt="image" src="https://github.com/user-attachments/assets/00802267-e1bd-4e04-8542-4b2eb045c39e" />

```
data["Sex"]=data["Sex"].astype("category")
data["Cabin"]=data["Cabin"].astype("category")
data["Embarked"]=data["Embarked"].astype("category")
data["Sex"]=data["Sex"].cat.codes
data["Cabin"]=data["Cabin"].cat.codes
data["Embarked"]=data["Embarked"].cat.codes
data
```

<img width="842" height="511" alt="image" src="https://github.com/user-attachments/assets/ae82974b-d7e0-4a33-8bfa-2b45f3d41fb1" />

```
x=pd.get_dummies(x)
k = 5
selector = SelectKBest(score_func=chi2, k=k)
x_new = selector.fit_transform(x,y)
selected_feature_indices = selector.get_support(indices=True)
selected_features = x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```

<img width="575" height="43" alt="image" src="https://github.com/user-attachments/assets/a73bada9-b1e9-4699-befd-1a77d840f322" />

```
x.info()
```

<img width="349" height="90" alt="image" src="https://github.com/user-attachments/assets/bb3be7fc-413f-41bb-bb8a-893730e87ce8" />

```
x=x.drop(["Name","Sex","Ticket","Cabin","Embarked"],axis=1,errors='ignore')
print(x.isnull().sum())
x = x.fillna(x.mean(numeric_only=True))
from sklearn.feature_selection import SelectKBest,f_regression
selector = SelectKBest(score_func=f_regression,k=5)
x_new = selector.fit_transform(x,y)
```

<img width="185" height="182" alt="image" src="https://github.com/user-attachments/assets/e1dbe49a-96c3-4b35-9866-5ea2e81e760e" />

```
selected_feature_indices = selector.get_support(indices=True)
selected_features = x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```

<img width="578" height="43" alt="image" src="https://github.com/user-attachments/assets/ad61b73e-49cd-46bd-aac5-894a98e91824" />

```
from sklearn.feature_selection import SelectKBest,mutual_info_classif
selector = SelectKBest(score_func=f_regression,k=5)
x_new = selector.fit_transform(x,y)
selected_feature_indices = selector.get_support(indices=True)
selected_features = x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```

<img width="553" height="38" alt="image" src="https://github.com/user-attachments/assets/bb429a61-167a-475b-91c8-370694bb522b" />

```
from sklearn.feature_selection import SelectPercentile,chi2
selector =  SelectPercentile(score_func=chi2,percentile=10)
x_new = selector.fit_transform(x,y)
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
sfm= SelectFromModel(model,threshold='mean')
sfm.fit(x,y)
```

<img width="546" height="226" alt="image" src="https://github.com/user-attachments/assets/7dfd6483-e8d3-4336-8cd9-1fc0c4bb9d94" />

```
selected_features = x.columns[sfm.get_support()]
print("Selected Features:")
print(selected_features)
```

<img width="470" height="44" alt="image" src="https://github.com/user-attachments/assets/24659b5f-fb8e-4030-b8a6-c78341d74829" />

```
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(x,y)
feature_importances = model.feature_importances_
threshold = 0.15
selected_features = x.columns[feature_importances > threshold]
print("Selected Features:")
print(selected_features)
```

<img width="473" height="38" alt="image" src="https://github.com/user-attachments/assets/fdc0c704-0f4d-42b2-8fef-60b83fb68545" />

```
df=pd.read_csv('titanic_dataset.csv')
df.columns
```

<img width="538" height="54" alt="image" src="https://github.com/user-attachments/assets/b4ab176a-9cae-4fe1-ad01-fe417c3ff1e4" />

```
df
```

<img width="825" height="609" alt="image" src="https://github.com/user-attachments/assets/7ee064b5-4da1-4c8f-8577-bdf1bb6849ba" />

```
df = df.dropna()
df.isnull().sum()
```

<img width="142" height="220" alt="image" src="https://github.com/user-attachments/assets/367e2eee-337c-439b-b51f-81830ceedcca" />

```
x = df[['PassengerId','Pclass','Age','SibSp','Parch','Fare']]
y = df['Survived']
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(score_func=f_classif,k=4)
x_new = selector.fit_transform(x,y)
x_new = selector.fit_transform(x, y)
selected_features_indices = selector.get_support(indices=True)
selected_features = x.columns[selected_features_indices]
print("Selected Features:")
print(selected_features)
```

<img width="465" height="44" alt="image" src="https://github.com/user-attachments/assets/aa5ddc65-e385-45a6-a004-593012a0750e" />

```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips= sns.load_dataset('tips')
tips.head()
```

<img width="382" height="189" alt="image" src="https://github.com/user-attachments/assets/44a668bb-50ea-438d-8182-a3acd703800e" />

```
contingency_table = pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```

<img width="184" height="77" alt="image" src="https://github.com/user-attachments/assets/7f09cb7e-5401-4834-96ab-bf2d9c0b2f03" />

```
chi2, p, _, _ = chi2_contingency(contingency_table)
print(f"Chi-Square Statistic: {chi2}")
print(f"P-value: {p}")
```

<img width="309" height="42" alt="image" src="https://github.com/user-attachments/assets/d3a01988-ee05-41c9-b607-ae278ee3e1f4" />

```
import pandas as pd
from sklearn.feature_selection import SelectKBest,mutual_info_classif,f_classif

data = {
    'Feature1': [1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3': [0,1,1,0,1],
    'Target': [0,1,1,0,1]
}
df = pd.DataFrame(data)
x = df[['Feature1','Feature3']]
y = df['Target']
import warnings
warnings.filterwarnings("ignore")
selector = SelectKBest(score_func=mutual_info_classif,k=1)
x_new = selector.fit_transform(x,y)
selected_features_indices = selector.get_support(indices=True)
selected_features = x.columns[selected_features_indices]
print("Selected Features:")
print(selected_features)
```

<img width="272" height="42" alt="image" src="https://github.com/user-attachments/assets/381a1e1c-cb5a-47fe-9af3-e7692afe90f8" />


# RESULT:
       # INCLUDE YOUR RESULT HERE
