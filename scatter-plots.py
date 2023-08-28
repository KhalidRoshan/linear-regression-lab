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
Scatter plots allow us to visualize the relationship between two variables (i.e., columns in
the dataframe)
'''

'''
Here we do a scatter plot for the words 'deal' and 'enron'
'''
y = df['enron'].to_numpy()
x = df['deal'].to_numpy()

# Sort X_test and y_pred for proper plotting
sorted_indexes = np.argsort(x)
sorted_x = x[sorted_indexes]
sorted_y = y[sorted_indexes]


# Plot the scatter plot of data points
plt.scatter(sorted_x, sorted_y, label='Scatter Plot')
plt.xlabel('enron')
plt.ylabel('deal')
plt.title('Scatter Plot Example')
plt.legend()
plt.show()