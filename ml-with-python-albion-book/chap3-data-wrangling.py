import pandas as pd
import numpy as np

from pandas.conftest import axis_1
from tensorflow.python.ops.logging_ops import Print

#  read csv
url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'
df = pd.read_csv(url)
df.head(5)

# 3.1 Creating a dataframe

dictionary = {
    "Name": ['Jacky Jackson', 'Steven Stevenson']
    ,"Age": [38, 25]
    ,"Driver": [True, False]
}
df = pd.DataFrame(dictionary)
# # add columns using a list of values

df["Eyes"] = ["Brown", "Blue"]

# 3.2 Getting information about the dataframe
url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'
df = pd.read_csv(url)
df.head(2)
## show dimensions
df.shape
## descriptive statistics for numeric cols
df.describe()
## show info
df.info()
# 3.3 Slicing dataframes

## select first row
df.iloc[0]
## select 3 rows
df.iloc[1:4]
## select 4 rows
df.iloc[:4]

# 3.4 Selecting rows based on conditionals

df[df['Sex'] == 'female'].tail(2)
## filter rows
df[(df['Sex'] == 'female') & (df['Age'] >= 65)]

# 3.5 Sorting values

df.sort_values(by = ['Age']).head(2)

# 3.6 Replacing values

df['Sex'].replace("female", "woman").head(2)
## replace multiple values at a go

df['Sex'].replace(["female", "male"], ["woman", "man"]).head(5)

## replace using regular expressions
df.replace(r"1st", "First", regex=True).head(2)

# 3.7 Renaming columns

df.rename(columns={'PClass': 'Passenger Class'}).head(2)

# 3.8 Summary statistics

print('Maximum', df['Age'].max())
print('Count', df['Age'].count())

## applying to the whole dataframe

df.count()

# 3.9 Finding unique values

df['Sex'].unique()

## count unique values
df['Sex'].value_counts()
df['PClass'].value_counts()

# 3.10 Handling missing values
## select missing values
df[df['Age'].isnull()].head()
df[df["Age"].isna()].head()

# 3.11 Deleting a column
## axis=1 specifies column
df.drop('Age', axis=1).head()
# if a column does not have a name you can
# delete it by its column index

df.drop(df.columns[1], axis=1).head(1)
# 3.12 Deleting a row

df[df['Sex'] != 'male'].head()

# 3.13 Dropping duplicate rows
## drop_duplicates drops rows matching perfectly across all columns
df.drop_duplicates().head(2)
## drop duplicates for a specified column
df.drop_duplicates(subset=['Sex', 'Age'])

# 3.14 Grouping rows by values

df.groupby('Sex').mean(numeric_only=True)
## group by second column
df.groupby(['Sex', 'Survived'])['Age'].mean()

# 3.15 Grouping rows by Time

## create date range
time_index = pd.date_range('06/06/2017', periods=100000, freq='30s')
## create dataframe
df = pd.DataFrame(index = time_index)
## create column of random values
df['Sale_Amount'] = np.random.randint(1, 10, 100000)
## group rows by week, calc sum per week
df.resample('W').sum()
## group by month, count rows
df.resample('ME').count()
# 3.16 Aggregating operations and statistics

## do an operation over each column
url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'
df = pd.read_csv(url)
df.agg('min')
## apply specific function to specific sets of columns
df.agg({"Age": ["mean"], "SexCode": ["min", "max"]})
## aggregate across groups
df.groupby(["PClass", "Survived"]).agg({"Survived": ["count"]})
# 3.17 Looping over a Column
## print first two names
for name in df["Name"][:5]:
    print(name.upper())

## using list comprehensions
[name.upper() for name in df['Name'][:5]]

# 3.18 Applying a Function over all Elemants in a columns

def uppercase(x):
    return x.upper()
df['Name'].apply(uppercase)[:2]

# 3.19 Applying functions to Groups

df.groupby('Sex').apply(lambda i: i.count())