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
Our objective is to explore the relationships between different variables (i.e., columns). 
The first step for this is to calculate their correlations. The corr() functions returns a matrix, where
every element represents the correlations between two variables. 
The elements in the diagonal represent the self-correlation, which is always 1.  
'''
df = df.drop('Email No.', axis=1) #We first drop the email number
#correlation_matrix = df.corr() # This may take a few minutes due to the large size of the
# dataset. You can uncomment he following lines to save the matrix and then load it when needed
# instead of recomputing everytime. Don't forget to comment the line above.   
#correlation_matrix.to_csv('correlations.csv')
correlation_matrix = pd.read_csv('correlations.csv', index_col=0)

'''
Since we have thousands of variables, we cannot simply look at the matrix. We instead 
calcualte the top and bottom, say 50, correlations. 
'''

# Get the absolute values of correlations
absolute_correlations = correlation_matrix.abs()

# Convert the matrix into a one column vector. Pandas will put the matrix values into a single column. 
# It will preserve their position by creating two index columns. One for the column and one for the row
# of the original value. 
correlation_series = absolute_correlations.unstack()
print(correlation_series)
input()

# Sort the correlation Series in descending order
sorted_correlations = correlation_series.sort_values(ascending=False)

# Filter out correlations with value 1 (self-correlations)
sorted_correlations = sorted_correlations[sorted_correlations != 1.0]

# Get the top 50 correlations
n = 50
top_n_correlations = sorted_correlations.head(n)

# Get the top 50 correlations
n = 50
bottom_n_correlations = sorted_correlations.tail(n)

print("Top", n, "correlated variables:")
print(top_n_correlations)

print("\n Bottom", n, "correlated variables:")
print(bottom_n_correlations)
