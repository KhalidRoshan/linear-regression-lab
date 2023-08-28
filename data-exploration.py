import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

''' 
Load the dataset. We use the Pandas library to read the comma separated value (CSV) file into a
dataframe.  A dataframe is the datastructure that Pandas uses to hold the data. You can roughly
think about it as a Excel sheet but for Python
'''

df = pd.read_csv('emails.csv')

''' 
After loading the dataset, we want to get an idea of its contents. Dataframes have a few useful
methods for this
'''

'''
The first method is to simply call the dataframe and print its contents
However, the output could be too busy to be useful.
'''
print(df)
input()

'''
Next, we try the info and describe methods. 
They will provide the size of the dataframe as well as some statistics about their contents. 
'''
print(df.info)
print(df.describe())
input()

'''
We also want to verify that the dataset is clean. What this means in this scenarion is that
there are no missing values. If there are any, we may need to fill them in.'''
print(df.isna())
input()

'''
Similarly to printing the whole dataset, printing the
output may not be too useful. Instead, we count the number of numeric values per column and 
compare it to the total number of rows. If the number of rows is equal to the number of non-missing
values in the column, then there are no missing values.
'''
non_missing = df.count()
print(non_missing)
input()

filtered_df = non_missing[non_missing<df.shape[0]] 
# retrieve the rows in num_numeric that are less than the number of rows in the original dataframe. 

print("If you see an output after this, you have missing values")
print(filtered_df)
input()







