
---
layout: post
title: RandomForest를 사용한 Titanic 데이터 분석
---


# Advanced Datamining Assignment #8 - RandomForest

#### Data Dictionary

Variable	Definition	Key  
survival	Survival	0 = No, 1 = Yes  
pclass	Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd  
sex	Sex	 
Age	Age in years	 
sibsp	# of siblings / spouses aboard the Titanic	
parch	# of parents / children aboard the Titanic	
ticket	Ticket number	
fare	Passenger fare	
cabin	Cabin number	
embarked	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton  

#### Variable Notes

pclass: A proxy for socio-economic status (SES)  
1st = Upper  
2nd = Middle  
3rd = Lower  
  
age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5  
  
sibsp: The dataset defines family relations in this way...  
Sibling = brother, sister, stepbrother, stepsister  
Spouse = husband, wife (mistresses and fiancés were ignored)  
  
parch: The dataset defines family relations in this way...  
Parent = mother, father  
Child = daughter, son, stepdaughter, stepson  
Some children travelled only with a nanny, therefore parch=0 for them.

# 1.Read data


```python
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix

os.chdir('D:/UNIST/17-2학기 자료/DM')

train=pd.read_csv('train.csv',index_col='PassengerId')
X_test=pd.read_csv('test.csv',index_col='PassengerId')
y_test=pd.read_csv('gender_submission.csv',index_col='PassengerId')
```

# 2. EDA


```python
# check data shape
train.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
    <tr>
      <th>PassengerId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
# check target variable
train.Survived.value_counts()
```




    0    549
    1    342
    Name: Survived, dtype: int64




```python
# summary train set
train.describe()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>714.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.383838</td>
      <td>2.308642</td>
      <td>29.699118</td>
      <td>0.523008</td>
      <td>0.381594</td>
      <td>32.204208</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.486592</td>
      <td>0.836071</td>
      <td>14.526497</td>
      <td>1.102743</td>
      <td>0.806057</td>
      <td>49.693429</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.420000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>20.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.910400</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.454200</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>38.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>31.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>80.000000</td>
      <td>8.000000</td>
      <td>6.000000</td>
      <td>512.329200</td>
    </tr>
  </tbody>
</table>
</div>




```python
# plot
plt.figure(figsize=(10, 5))

plt.subplot(1,2,1)
sns.countplot('Sex',hue='Survived',data=train)
plt.title('Sex:Survived vs Dead')

plt.subplot(1,2,2)
sns.countplot('Pclass',hue='Survived',data=train)
plt.title('Pclass:Survived vs Dead')

plt.show()
```


![png](output_8_0.png)



```python
# pclass vs survived
pd.crosstab(train.Pclass,train.Survived,margins=True) #.style.background_gradient()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Survived</th>
      <th>0</th>
      <th>1</th>
      <th>All</th>
    </tr>
    <tr>
      <th>Pclass</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>80</td>
      <td>136</td>
      <td>216</td>
    </tr>
    <tr>
      <th>2</th>
      <td>97</td>
      <td>87</td>
      <td>184</td>
    </tr>
    <tr>
      <th>3</th>
      <td>372</td>
      <td>119</td>
      <td>491</td>
    </tr>
    <tr>
      <th>All</th>
      <td>549</td>
      <td>342</td>
      <td>891</td>
    </tr>
  </tbody>
</table>
</div>



### Missing value Imputaion


```python
# check null value
train.isnull().sum()
```




    Survived      0
    Pclass        0
    Name          0
    Sex           0
    Age         177
    SibSp         0
    Parch         0
    Ticket        0
    Fare          0
    Cabin       687
    Embarked      2
    dtype: int64




```python
train.Name.head()
```




    PassengerId
    1                              Braund, Mr. Owen Harris
    2    Cumings, Mrs. John Bradley (Florence Briggs Th...
    3                               Heikkinen, Miss. Laina
    4         Futrelle, Mrs. Jacques Heath (Lily May Peel)
    5                             Allen, Mr. William Henry
    Name: Name, dtype: object



#### Embarked


```python
train.Embarked.value_counts()
```




    S    644
    C    168
    Q     77
    Name: Embarked, dtype: int64



#### Fares


```python
plt.figure(figsize=(10, 5))

plt.subplot(1,3,1)
sns.distplot(train[train['Pclass']==1].Fare)
plt.title('Fares in Pclass 1')

plt.subplot(1,3,2)
sns.distplot(train[train['Pclass']==2].Fare)
plt.title('Fares in Pclass 2')

plt.subplot(1,3,3)
sns.distplot(train[train['Pclass']==3].Fare)
plt.title('Fares in Pclass 3')
plt.show()
```


![png](output_16_0.png)



```python
# get Initial function
def get_Initial(data):
    # using regular expression
    data['Initial']=data.Name.str.extract('([a-zA-Z]+)\.',expand=True)
    return pd.crosstab(data.Sex,data.Initial)

# Imputation function
def treat_missing_value(data):
    new_list=[]
    for initial in data.Initial:
        if initial in ['Miss','Mile','Mme','Ms']:
            initial='Miss'
        elif initial in ['Mr','Dr','Major','Capt','Sir','Don']:
            initial='Mr'
        elif initial in ['Mrs','Lady']:
            initial='Mrs'
        elif initial in ['Master']:
            initial='Master'
        else:
            initial='others'
        new_list.append(initial)
    data.Initial=new_list
    
    # Age imputation group by Initials
    result=dict(data.groupby('Initial')['Age'].mean().round(0))
    initials=['Mr','Mrs','Master','Miss','Other']
    for i in initials:
        data.loc[(data.Age.isnull())&(data.Initial==i),'Age']=result.get(i)
    
    # Embarked imputation
    data['Embarked'].fillna('S',inplace=True)
    
    # Fare imputation group by Pclass
    result=dict(train.groupby('Pclass').mean().Fare.round(0))
    for i in [1,2,3]:
        data.loc[(data.Fare.isnull())&(data.Pclass==i),'Fare']=result.get(i)
        
    return
```


```python
# apply 
get_Initial(train)
treat_missing_value(train)

get_Initial(X_test)
treat_missing_value(X_test)
```

## new variable


```python
# add SibSp, Parch
train['Family_num']=train.SibSp+train.Parch

X_test['Family_num']=X_test.SibSp+X_test.Parch
```


```python
# plot
sns.countplot('Family_num',hue='Survived',data=train)
plt.title('Family number vs Survived')

plt.show()
```


![png](output_21_0.png)


### Correlation


```python
# correlation matrix
sns.heatmap(train.corr(),annot=True,linewidths=0.2)
fig.set_size_inches(10,8)
plt.show()
```


![png](output_23_0.png)


#### Drop columns & dummy encoding


```python
# drop and get dummy function
def drop_and_get_dummy(data):
    #drop columns
    data.drop(['Name','Ticket','Cabin'],axis=1,inplace=True)
    
    # dummy variable
    obj_col=['Sex','Embarked','Initial']
    data=pd.get_dummies(data,columns=obj_col,drop_first=True)

    return data
```


```python
train=drop_and_get_dummy(train)
X_test=drop_and_get_dummy(X_test)
```

# 3. modeling

### RandomForest


```python
# train -> split target variable
X_train,y_train=train.iloc[:,1:],train.Survived
```


```python
param_test = {
    'n_estimators':[10,100,1000,2000],
    'criterion':['gini','entropy']
}
gsearch = GridSearchCV(estimator = RandomForestClassifier(random_state= 42 ), 
         param_grid = param_test,scoring='accuracy', cv=10, n_jobs=4)

gsearch.fit(X_train,y_train)
gsearch.grid_scores_, gsearch.best_params_, gsearch.best_score_
```




    ([mean: 0.80135, std: 0.05487, params: {'criterion': 'gini', 'n_estimators': 10},
      mean: 0.80471, std: 0.04587, params: {'criterion': 'gini', 'n_estimators': 100},
      mean: 0.80808, std: 0.04646, params: {'criterion': 'gini', 'n_estimators': 1000},
      mean: 0.80696, std: 0.04629, params: {'criterion': 'gini', 'n_estimators': 2000},
      mean: 0.80359, std: 0.03922, params: {'criterion': 'entropy', 'n_estimators': 10},
      mean: 0.80584, std: 0.04391, params: {'criterion': 'entropy', 'n_estimators': 100},
      mean: 0.80696, std: 0.04836, params: {'criterion': 'entropy', 'n_estimators': 1000},
      mean: 0.80808, std: 0.04676, params: {'criterion': 'entropy', 'n_estimators': 2000}],
     {'criterion': 'gini', 'n_estimators': 1000},
     0.8080808080808081)




```python
# modeling with the results of gridSearch
criterion=gsearch.best_params_['criterion']
n_estimators=gsearch.best_params_['n_estimators']
rf=RandomForestClassifier(criterion=criterion,n_estimators=n_estimators)
rf.fit(X_train,y_train)
```




    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=1000, n_jobs=1,
                oob_score=False, random_state=None, verbose=0,
                warm_start=False)




```python
# confusion matrix
pred_rf=rf.predict(X_test)

print(confusion_matrix(y_test,pred_rf))
print(classification_report(y_test,pred_rf, 
                            target_names=['Dead', 'Survived']))
```

    [[217  49]
     [ 35 117]]
                 precision    recall  f1-score   support
    
           Dead       0.86      0.82      0.84       266
       Survived       0.70      0.77      0.74       152
    
    avg / total       0.80      0.80      0.80       418
    
    

## 4. Summary of analysis

Sex: The chance of survival for women is high as compared to men.

Pclass:There is a visible trend that being a 1st class passenger gives you better chances of survival.   
       The survival rate for Pclass3 is very low. 
      
Parch+SibSp: Having 1-2 siblings,spouse on board or 1-3 Parents shows a greater chance of probablity rather than being alone or having a large family travelling with you.
  
We could predict about 80% dead / survived with 1000 random forest estimators.
