import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


''' 
Load the dataset. We use the Pandas library to read the comma separated value (CSV) file into a
dataframe.  A dataframe is the datastructure that Pandas uses to hold the data. You can roughly
think about it as a Excel sheet but for Python
'''
df = pd.read_csv('emails.csv')
#df=df.sample(n=100,  random_state=1)
columns_to_drop = df.columns[df.eq(0).all()]
df = df.drop(columns=columns_to_drop)

'''
In this script, we will use multi-linear regression to predict the number of occurrances of a certain
word based on all other word ocurrances. That is, we will choose one column of the dataset as the target
 and use the rest of the columns to predict it.
'''

'''
There are two special columns in this dataset that we need to remove. The 'Email no.' columns and 
the 'Prediction' column.
'''
df = df.drop(columns=['Email No.', 'Prediction'])
'''
It is always a good idea to normalize the values in the dataset as they may be in very different ragnes. 
This can affect the numerical stability of the traiing algorithm. 
'''
# df = (df - df.min()) / (df.max() - df.min())

'''
Next, we choose the column that we want to predict. For this example, I am choosing the column for
'deal' which is column. This column will become the target data that we are predicting.
'''
target_column = 'deal'
y = df[target_column]
y = y.to_numpy()

'''
Our training dataset X is all the columns left in dataframe except the target column 
'''
X = df.drop(columns=[target_column])
#X = X.sample(n=10, axis=1, random_state=9)# For debugging, we can choose a small random set of columns. 
X = X.to_numpy()

'''
We are now ready to split the data into training and testing sets. Here, we use 20% of the data for
testing and set a seed for the pseudorandom number generator so that we always get the same training and
testing sets. 
'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

'''
The next few steps are the routine steps that we follow to train a model and obtain the prediction
results. 
'''
# Initialize the LinearRegression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the mean squared error to evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)

print("Coefficient:", model.coef_) #w's in the formulas
print("Intercept:", model.intercept_)# b in the formulas
print("Mean Squared Error:", mse)

'''
Since we have many dimensions, we can only plot one. Here, I am arbitrarly choosing the fifth
column. However, you can choose other columns and see how the plots change. 
'''
# Sort X_test and y_pred for proper plotting
X_plot = X_test[:, 4]
sorted_indexes = np.argsort(X_plot)
sorted_X_test = X_plot[sorted_indexes]
sorted_y_pred = y_pred[sorted_indexes]
sorted_y_test = y_test[sorted_indexes]


# Plot the scatter plot of data points
plt.scatter(sorted_X_test, sorted_y_test, label='Test Data')
#plt.scatter(X_test[:,test_loc], y_test, label='Test Data')

'''
Note that we are no longer using plot(). We are using scatter(). The reason is that we no longer
have a nice straight line since the prediction depends on many variables and we are only plotting one. 
If we limite the number of columns to two, it is possible to plot the straight line in 3D. 
'''
# Plot the linear regression line
plt.scatter(sorted_X_test.squeeze(), sorted_y_pred, color='red', label='Linear Regression', marker='o')

plt.xlabel('X')
plt.ylabel('y')
plt.title('Multi-Linear Regression Example')
plt.legend()
plt.show()