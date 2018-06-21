---
layout: post
title: "Edge Case: Nested and Mixed Lists"
categories:
  - Edge Case
tags:
  - content
  - css
  - edge case
  - lists
  - markup
last_modified_at: 2017-03-09T14:25:52-05:00
---

Nested and mixed lists are an interesting beast. It's a corner case to make sure that lists within lists do not break the ordered list numbering order and list styles go deep enough.

## Ordered -- Unordered -- Ordered

1. ordered item
2. ordered item
  * **unordered**
  * **unordered**
    1. ordered item
    2. ordered item
3. ordered item
4. ordered item

## Ordered -- Unordered -- Unordered

1. ordered item
2. ordered item
  * **unordered**
  * **unordered**
    * unordered item
    * unordered item
3. ordered item
4. ordered item

## Unordered -- Ordered -- Unordered

* unordered item
* unordered item
  1. ordered
  2. ordered
    * unordered item
    * unordered item
* unordered item
* unordered item

## Unordered -- Unordered -- Ordered

* unordered item
* unordered item
  * unordered
  * unordered
    1. **ordered item**
    2. **ordered item**
* unordered item
* unordered item


#

# coding: utf-8

# ## read

# In[19]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.chdir('C:/Users/poscouser/Downloads/')

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


plt.rc("font", size=14)
sns.set(style="whitegrid", color_codes=True)

# matplotlib 한글 깨짐처리
from matplotlib import font_manager, rc
font_url = "C:/Users/poscouser/Downloads/malgun.ttf"

font_name = font_manager.FontProperties(fname=font_url).get_name()
rc('font', family=font_name)


# In[15]:


# data = pd.read_csv("leader_wo_researches_bin2.csv", header=0)
data = pd.read_excel("leader_wo_researches_bin21.xlsx")


# ## new data import

# In[ ]:


j=pd.read_excel('그룹장_모델용데이터_v3.xlsx')
l=pd.read_excel('팀리더_모델용데이터_v3.xlsx')


# In[54]:


for k,col in enumerate(l.columns):
    if col not in j.columns:
        print(k,col in j.columns)


# In[24]:


test1=pd.read_excel('후보자_그룹장_20180618_v3.xlsx')
test2=pd.read_excel('후보자_팀리더_20180618_v3.xlsx')


# ###### 변수 선택

# In[17]:


# 선택할 변수 리스트
data_corr1 = ['Class', ### 'EMPLOYEE_ID', 
                   '유학총회수',
                   '국내비학위과정', '국내산학박사과정', '국내산학석사과정', '국내산학학사과정',
                   '국내석사과정', '국내정규박사과정', '국내정규석사과정', '해외산학석사과정', 
                   '해외연구과정', '해외정규박사과정', '해외정규석사과정', '해외21C챌린저과정', ## '해외석사과정', 
                   
                   'EVAL_AVG', 'COMPETENCE_POINT_역량평가',  'SELF_POINT_자기평가','Gap', ##'BB이상 보유갯수', '업적평가점수',
                   #'승진시나이', 
                   '징계상태여부', ### 'HANDICAP_GRADE_NM', '장애코드', 
                   'Keyjob요원(누계)', ##   'Keyjob요원',
                   #pcp
                   'Norm_MONTH2', ### '해당년도 누계',
#                    '결혼여부', '성별', '표_자녀수',
                   ##포상
                   '개수',  ### '개수(누계)', 
                   ## PPMS
                   '수행개수(누계)',   #### '개수(해당년_종료)',
                   '년도별 봉사시간', 
                   '입사전경력수',  ### '입사전경력년수',
                   '포항','광양', '부산','서울', '송도', '순천','기타', ## '광주', '대전', '해외',
                   '파견기간', ###  '파견회수(누계)', 
                   '입사유형',
                   '자격증개수', '최다성격', 
                   '표_입사학교전공명', #'최종학위명','표_입사학교명',  
                   #'표_병역구분명(병역)', '해외근무중', 
                   '자격_BB', '자격_CPSM', '자격_기사', '자격_기술사', '자격_노무사', '자격_변호사', '자격_회계사',
                   '해외근무_유무',
                   '해외근무_남아프리카공화국', '해외근무_뉴칼레도니아', '해외근무_독일', '해외근무_러시아', 
                   '해외근무_말레이시아', '해외근무_멕시코', '해외근무_몽골', '해외근무_미국', '해외근무_미얀마', 
                   '해외근무_베네수엘라', '해외근무_베트남', '해외근무_벨기에', '해외근무_브라질', '해외근무_슬로바키아',
                   '해외근무_슬로베니아', '해외근무_싱가포르', '해외근무_아랍에미리트', '해외근무_아르헨티나', '해외근무_우루과이',
                   '해외근무_이집트', '해외근무_이탈리아', '해외근무_인도네시아', '해외근무_일본',
                   '해외근무_중국', '해외근무_체코', '해외근무_칠레', '해외근무_카자흐스탄', '해외근무_캐나다',
                   '해외근무_태국', '해외근무_터키', '해외근무_파키스탄', '해외근무_폴란드', '해외근무_필리핀', 
                   '해외근무_호주', '해외근무_홍콩', ### '해외근무_이란', '해외근무_프랑스', 
                   
                   '언어_표준등급', 
                   '중국어', '러시아어', '베트남어', '스페인어', '아랍어', '프랑스어', 
                   '포르투갈어',  '이탈리아어', '인도네시아어', '태국어', '일본어',#'독일어',, '힌두어' '터키어', 
                   #"출신시도" 
]

data=data.loc[:,data_corr1]


# In[18]:


data.isna().sum()


# In[398]:


def na_delete(data):    
    data['언어_표준등급'] =data['언어_표준등급'].fillna(value="없음")
    data=data.fillna(0)
    return data


# In[356]:


data.loc[:,'중국어':]=data.loc[:,'중국어':].fillna(0)
data['언어_표준등급'] =data['언어_표준등급'].fillna(value="없음")
data.loc[:,:'해외근무_홍콩']=data.loc[:,:'해외근무_홍콩'].fillna(0)


# In[357]:


# 
def column_select(data,percent):    
    new_col=[]
    for col in data.columns:
        data_shape=data.shape[0]
        most_class=data.loc[:,col].value_counts()[0]
        if most_class<data_shape*percent:
            new_col.append(col)
    return new_col


# In[358]:


# % 기준으로 변수 제거
new_col=column_select(data,0.95)


# In[359]:


# select columns
new_col


# # DF

# #### dummy

# In[360]:


df=data.loc[:,new_col]


# In[330]:


def language_preprocessing(df):
    # level selection
    df['level']=df.언어_표준등급.apply(lambda x:x[-1])
    level_dict=dict({'A':3,'B':2,'C':1,'음':0})
    df.level=df.level.map(level_dict)

    # 언어 selection
    df.언어_표준등급=df.언어_표준등급.apply(lambda x: x[:-1])
    
    return df


# In[361]:


df=language_preprocessing(df)


# In[120]:


# level selection
df['level']=df.언어_표준등급.apply(lambda x:x[-1])
level_dict=dict({'A':3,'B':2,'C':1,'음':0})
df.level=df.level.map(level_dict)

# 언어 selection
df.언어_표준등급=df.언어_표준등급.apply(lambda x: x[:-1])

language_variation=len(df.언어_표준등급.value_counts())-1
print('유니크 언어 개수 :',language_variation)


# In[331]:


def dummy_coding(df):
    # category 변수 dummy 
    cat_vars = ["입사유형","최다성격", "표_입사학교전공명", "언어_표준등급"]

    df=pd.get_dummies(df,columns=cat_vars)
    df=df.drop(labels=['언어_표준등급_없','최다성격_0'],axis=1)

    # 더미 변수화 한 언어에 level 반영하기
    lan_col=df.columns[-language_variation:]

    for col in lan_col:
        df.loc[:,col]=df.loc[:,col]*df.level

    df=df.drop(labels=['level'],axis=1)
    print('dataset shape :',df.shape)
    print('NA check :' ,sum(df.isna().sum()))

    return df


# In[378]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder


# In[121]:


# category 변수 dummy 
cat_vars = ["입사유형","최다성격", "표_입사학교전공명", "언어_표준등급"]

df=pd.get_dummies(df,columns=cat_vars)
df=df.drop(labels=['언어_표준등급_없','최다성격_0'],axis=1)


# In[122]:


# 더미 변수화 한 언어에 level 반영하기
lan_col=df.columns[-language_variation:]

for col in lan_col:
    df.loc[:,col]=df.loc[:,col]*df.level

df=df.drop(labels=['level'],axis=1)
print('dataset shape :',df.shape)
print('NA check :' ,sum(df.isna().sum()))


# #### correlation check

# In[123]:


df.columns


# In[64]:


# Compute the correlation matrix
corr = np.round(df.iloc[:,:25].corr(),2)

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(15, 15))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,annot=True,
            square=True, linewidths=.5, cbar_kws={"shrink": .5},)


# # Modeling

# In[70]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# #### model based variable selection (lasso)

# In[124]:


logreg=LogisticRegression(random_state=42,penalty='l1')
#rf=RandomForestClassifier(random_state=42)

def multiple_rfe(train_data,train_target,model,k):
    rfe=RFE(model,k)
    rfe.fit(train_data,train_target)
    return list(train_data.iloc[:,rfe.support_].columns)


# data frame 형태로 traindata, train target, 선택 할 변수 개수 입력 하시면 됩니다.
def select_variables(train_data,train_target,k):
    variables=[]
    for i in models:
        variables.append(multiple_rfe(train_data,train_target,i,k))
    freq_variables=pd.DataFrame(variables).unstack().reset_index().iloc[:,-1]
    return list(freq_variables.value_counts()[:k].index)


# In[128]:


# logistic 기반 변수 선택 코드 입니다
def get_logistic_coefficient(train_data,model2):
    coef=pd.DataFrame(np.transpose(model2.coef_))
    coef.index=train_data.columns
    coef= coef[(coef.T != 0).any()]
    coef=round(coef,2)
    return coef

def variable_selection_logistic(train_data,train_target,model2):
    model2.fit(train_data,train_target)
    # select variables (logistic regression l1 norm)
    selected=get_logistic_coefficient(train_data,model2).index
    print('# of select variables :',len(selected))
    return selected


# In[160]:


# split data
X_train,X_test,y_train,y_test=train_test_split(df.iloc[:,1:],df.Class,random_state=42, stratify=df.Class)


model= LogisticRegression(C=0.2, penalty="l1",class_weight='balanced')


selected_variables=variable_selection_logistic(X_train,y_train,model)

X_train=X_train.loc[:,selected_variables]
X_test=X_test.loc[:,selected_variables]


# In[389]:


fin_col=X_train.columns


# #### stat model results

# In[163]:


import statsmodels.api as sm

stat=sm.Logit(y_train,X_train)

results=stat.fit()
results.summary2()
y_pred = results.predict(X_test)


# In[329]:


plt.scatter(y_pred,y_test)


# In[300]:


def stat_results(y_pred,threshold):
    
    pred=np.where(y_pred>threshold,1,0)
    print('statsmodel results')
    print(classification_report(y_test,pred,digits=3))    
    
    category_list=['비대상','대상']
    ct=pd.crosstab(pred,y_test)
    ct.columns=category_list
    ct.index=category_list
    print('cross table \n\n',ct)


# In[264]:


y_pred = results.predict(X_test)

stat_results(y_pred,0.5)


# #### input example

# In[ ]:


### 데이터 불러오기


# In[454]:


test=data.head(30)


# In[ ]:


### 변수명 맞춰주기


# In[455]:


def preprocessing_all(test):
    test=test.loc[:,new_col]
    test=na_delete(test)
    test=language_preprocessing(test)
    test=dummy_coding(test)
    test=test.reindex(columns=fin_col).fillna(0)
    return test


# In[456]:


test=preprocessing_all(test)


# In[457]:


# new data input example
def new_data_predict(test,threshold):
    new_pred = results.predict(test)
    
    pred=np.where(new_pred>threshold,1,0)
    
    r_df=pd.DataFrame({'INDEX':test.index,'Probability':new_pred,'Predict':pred}).reset_index(drop=True)
    return r_df


# In[458]:


new_data_predict(test,0.5)


# ## Grid search

# ##### 최종 변수 선택

# In[164]:


from sklearn.model_selection import GridSearchCV


# In[165]:


logreg= LogisticRegression(C=1.5, penalty="l1",class_weight='balanced')


# In[166]:


params={
    'C':[0.1, 0.5, 1, 1.5, 2],
}

def get_grid(model,params,scoring):
    grid = GridSearchCV(model, params, cv=5, verbose=0, n_jobs=4, scoring=scoring)
    grid.fit(X_train,y_train)

    print('grid.best_parameters :',grid.best_params_)
    print('\n\n grid best score :',grid.best_score_)
    print('\n\n grid resutls :',pd.DataFrame(grid.grid_scores_).iloc[:,:2])
    return grid.best_estimator_


# In[167]:


gd_model=get_grid(logreg,params,'accuracy')


# ## Model fitting

# In[192]:


def model_fit(x):
    x.fit(X_train,y_train)
    print(x.fit,'\n')
    
    # test pred
    pred=x.predict(X_test)
    report=classification_report(y_test,pred,digits=3)
    
    # train pred
    train_pred=x.predict(X_train)
    train_report=classification_report(y_train,train_pred,digits=3)
    
    print('train fit report : \n\n',train_report)
    print('test fit report : \n\n', report)
    
    category_list=['비대상','대상']
    ct=pd.crosstab(pred,y_test)
    ct.columns=category_list
    ct.index=category_list
    
    print('cross table \n\n',ct)
    return x


# In[219]:


# random forest
rf=RandomForestClassifier(n_estimators=30,random_state=42,class_weight='balanced')


# In[193]:


mm=model_fit(gd_model)


# In[220]:


mm=model_fit(rf)


# ##### Appendix
# vif check

from statsmodels.stats.outliers_influence import variance_inflation_factor    

def calculate_vif_(X, thresh=100):
    cols = X.columns
    variables = np.arange(X.shape[1])
    dropped=True
    while dropped:
        dropped=False
        c = X[cols[variables]].values
        vif = [variance_inflation_factor(c, ix) for ix in np.arange(c.shape[1])]

        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            print('dropping \'' + X[cols[variables]].columns[maxloc] + '\' at index: ' + str(maxloc))
            variables = np.delete(variables, maxloc)
            dropped=True

    print('Remaining variables:')
    print(X.columns[variables])
    return X[cols[variables]]

s=calculate_vif_(df,4)# p_value
selected_variables_p_check=results.pvalues.index[results.pvalues.values<=0.05]
print('# of variables p-value <=0.05 :',len(selected_variables_p_check))

X_train=X_train.loc[:,selected_variables_p_check]
X_test=X_test.loc[:,selected_variables_p_check]pd.DataFrame({'coef':mm.coef_[0],'variabes':X_train.columns})
