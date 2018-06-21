---
layout: post
title: "Edge Case: Many Tags"
categories:
  - Edge Case
tags:
  - 8BIT
  - alignment
  - Articles
  - captions
  - categories
  - chat
  - comments
  - content
  - css
  - dowork
  - edge case
  - embeds
  - excerpt
  - Fail
  - featured image
  - FTW
  - Fun
  - gallery
  - html
  - image
  - Jekyll
  - layout
  - link
  - Love
  - markup
  - Mothership
  - Must Read
  - Nailed It
  - Pictures
  - Post Formats
  - quote
  - standard
  - Success
  - Swagger
  - Tags
  - template
  - title
  - twitter
  - Unseen
  - video
  - YouTube
---

This post has many tags.





# coding: utf-8

# # TEXT

# In[1]:


import os,re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

os.chdir('C:/Users/poscouser/Downloads/MBO_복호화_완료')


# ### read data

# In[414]:


data_list=os.listdir()[1:-2]

def read_all():    
    df=pd.DataFrame()
    for i in data_list[0:9]:
        data=pd.read_excel(i)
        df=pd.concat([df,data])
    return df

print(data_list)


# In[571]:


df=read_all()


# In[572]:


df.isna().sum()


# In[573]:


# read synonym dictionary
sy=pd.read_excel('synonym_v1.xlsx')
sy.columns=['ff','tt']
sy_dict=dict(zip(sy.ff,sy.tt))

def noun_merge():
    # additional nouns 
    noun1=pd.read_excel('product_orgs_dict_v2.xlsx').iloc[:,0]
    noun2=pd.read_excel('POSCO_All_02.xlsx').iloc[:,0]

    noun2.append([noun1,sy.ff,sy.tt])
    return noun2


# In[574]:


# 사전 추가 단어 불러오기
nouns=noun_merge()


# #### cleaning

# In[576]:


def preprocessing(df):    
    df.EMPLOYEE_ID=df.EMPLOYEE_ID.fillna(0)+df.Employee_ID.fillna(0)
    df.Year=df.Year.fillna(0)+df.년도.fillna(0)
    df=df.drop(['Employee_ID','년도'],axis=1)
    df=df.reset_index(drop=True)
    
    # delete some na in EmployeeID
    reference_table=pd.read_excel('MBO_EmployeeID_join.xlsx')
    reference_table.직번=reference_table.직번.astype(str)
    ref_dict=dict(zip(reference_table.직번,reference_table.EMPLOYEE_ID))
    df.직번=df.직번.astype(str)

    df.EMPLOYEE_ID=df.EMPLOYEE_ID+df.직번.map(ref_dict).fillna(0)
    
    return df


# In[577]:


df=preprocessing(df)


# # eda

# In[588]:


# drop na employee id
df=df.query('EMPLOYEE_ID!=0')
df=df.loc[:,['EMPLOYEE_ID','목표명','상반기 업무실적','업무실적']].reset_index(drop=True)


# In[590]:


df.isna().sum()


# In[591]:


# na를 빈 값으로 만들고, 하나의 컬럼으로 만들기
def get_merged_text(df):
    df=df.fillna(' ')
    df['text']=df.목표명+' '+df.iloc[:,-2]+' '+df.iloc[:,-1]
    df=df.iloc[:,[0,-1]]
    
    df=df.groupby('EMPLOYEE_ID').text.apply(lambda x: "{%s}" % ', '.join(x)).reset_index()
    return df


# In[592]:


grouped_text=get_merged_text(df)


# ## Key word example

# In[609]:


def get_keyword(code):
    # search keyword data frame
    search_keyword_df=pd.read_excel('Pilot30_keywords.xlsx')
    search_keyword_df.columns=['POST명', 'POST_CODE', '순위', '필요경험', '키워드']
    code_dict=dict(zip(search_keyword_df.POST_CODE,search_keyword_df.POST명))
    
    keyword=search_keyword_df.query("POST_CODE=={0}".format(code)).키워드
    print('이 코드는 {0} 부서를 찾는 코드 입니다'.format(code_dict.get(code)))
    return keyword


# In[614]:


search_keyword_df.POST_CODE.unique()


# In[615]:


keyword=get_keyword(12521)


# #### tagging

# In[176]:


from ckonlpy.tag import Twitter

tw=Twitter()

# add dictionary
tw.add_dictionary(list(nouns), 'Noun')


# In[179]:


def tw_apply(text):
    results=tw.nouns(text)
    results=[noun for noun in results if len(noun)>1]
    return ', '.join(results)

def check_nouns(hr):
    h=hr.apply(tw_apply).apply(lambda x: x.split(', '))
    h=[j for i in h for j in i]
    return h


def get_noun_list_df(grouped_text):
    # 일부만 확인 
    sample=grouped_text.iloc[-500:,:] # 전체 사용시 아래 코드로
    #sample=grouped_text
    sample['nouns']=sample.text.apply(tw_apply)
    sample=sample.reset_index(drop=True)
    sample.nouns=sample.nouns.apply(lambda x: x.split(', '))

    print(sample.shape)
    return sample.loc[:,['EMPLOYEE_ID','nouns']]


# In[404]:


keyword=check_nouns(keyword)


# In[197]:


get_ipython().run_cell_magic('time', '', '\n# it takes some time... \nsample=get_noun_list_df(grouped_text)')


# In[236]:


def mapping(noun_list):
    result=pd.Series(noun_list).map(sy_dict)
    result=[noun_list[i] if np.isnan(result[i]) else i for i in range(len(result))]
    return result


# In[241]:


# synomyn mapping
sample.nouns=sample.nouns.apply(mapping)


# In[396]:


from collections import Counter


def check_df(keyword,sample):    
    '''
    keyword : check keyword list
    sample : employee_id 별로 텍스트가 있는 data frame 
    '''
    
    unique_check_list=list(set(keyword))
    unique_check_len=len(unique_check_list)
    
    results=[]
    for k in range(sample.shape[0]):
        c=Counter(sample.nouns[k])
        
        count_result=[c.get(i) for i in unique_check_list] 
        # results=[c.get(i) for i in unique_check_list]  first version
        results.append(list(pd.Series(count_result).fillna(0).apply(lambda x : 1 if x>0 else x)))
        
    df_result=pd.DataFrame(results,columns=unique_check_list).dropna(axis=1,how='all').fillna(0)
    
    # make new columns
    df_result['Total_sum']=np.sum(df_result,axis=1)
    df_result['Total_ratio']=np.round(df_result.Total_sum/unique_check_len,3)
    
    df_result=pd.concat([sample,df_result],axis=1)
    return df_result


# In[397]:


df_result=check_df(keyword,sample)


# In[ ]:


df_result


# #### appendix
from sklearn.feature_extraction.text import CountVectorizer

c=CountVectorizer()
# In[ ]:


df.EMPLOYEE_ID=df.EMPLOYEE_ID.fillna(0)
df.Employee_ID=df.Employee_ID.fillna(0)
df.Year=df.Year.fillna(0)
df.년도=df.년도.fillna(0)

#df.query('EMPLOYEE_ID!=0 & Employee_ID!=0')
#df.query('Year!=0 & 년도!=0')

# sample search keyword

hr=['인사기획', '글로벌HR기획','노무기획','인사운영','제철소 인사운영','경영전략 수립','경영전략수립','그룹전략 수립']

gr=['자재구매','설비구매','외주계약','구매기획','원료 내자구매','내자구매']

st=['인사제도 기획','인사제도','인사평가','채용','보직운영','채용제도','조직운영',
    '해외법인 관리','노무제도 기획','노무기획','급여제도','복리후생','E직군 인사']# 정규표현식을 사용한 노이즈 제거
def delete_noise(text):
    noise = re.compile('[\t\n\r\xa0]')                
    result=noise.sub('',str(text))                    #  \t, \n, \r, \xa0 제거 
    result=re.sub(' +<.*?>','',result)                # 특수문자 제거
    result=re.sub(r'[^\w]',' ',result)                # 특수문자 제거
    result=re.sub(' +',' ',result).strip()            # 여러 공백(multi space)을 하나의 공백으로 줄이기
    return result.split(' ')
