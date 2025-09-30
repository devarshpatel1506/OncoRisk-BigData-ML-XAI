import pandas as pd

df1=pd.read_csv('/Users/mayankkapadia/Desktop/SJSUMATERIAL/SEM2/Big_Data/project/bcsc_risk_factors_summarized1_092020.csv')
df2=pd.read_csv('/Users/mayankkapadia/Desktop/SJSUMATERIAL/SEM2/Big_Data/project/bcsc_risk_factors_summarized2_092020.csv')
df3=pd.read_csv('bcsc_risk_factors_summarized3_092020.csv')

df=pd.DataFrame()

df=pd.concat([df1,df2,df3],ignore_index=True)


#Checking any Null Values
print(df.isnull().any())

df=df.drop(columns=['count','year'])

print(df.head())

df.to_csv('final_risk_factor.csv',index=False)