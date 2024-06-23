import pandas as pd

# create the sample dataframe
df = pd.DataFrame({'Student_ID':[1,2,3,4,5], 
                   'High_School_Type':['State','Private','Other','State', 'State'],
                   'Letter_Grade':['A','C','F','B','D'],
                   'Outcome':['Passed','Passed','Failed','Passed','Failed']})

# Label encode Letter_Grade and Outcome columns
df['Letter_Grade'] = df['Letter_Grade'].replace(
    {'A':4, 
    'B':3, 
    'C':2, 
    'D':1, 
    'F':0})

df['Outcome'] = df['Outcome'].replace(
    {'Passed':1, 
     'Failed':0})

print(df.head())

# One-hot encode High_School_Type column
df = pd.get_dummies(
    df, 
    columns=['High_School_Type'], 
    dtype=int)

print(df.head())

