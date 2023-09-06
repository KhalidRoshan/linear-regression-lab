# linear-regression-lab

## Instructions
The objective of this lab is to familiarize you with linear regression. We will be using the 
spam email dataset based on the Enron case. You can find it in this repository and
also in the original post linked at the bottom. 

*Deliverable*: The deliverable is a report with the required screenshots and answers to the
questions in the instructions below. 

### Data exploration
The first step in any data science/ML/AI project is to take a look at the dataset. The objective is
two-fold. First, we need to know if there are any missing data points or if we have data in
unexpected formats. Second, we want to know basic statistical facts about the dataset including
number of rows and columns, maximum and minimum values, and the largest correlations between
columns. 

1. Run the file ```data-exploration.py``` on a terminal. You can do this by typing 
```python data-exploration.py```. Note that you cannot submit this file through the ```sbatch```
command due to the ```input()``` functions. 

2. Press enter to move through the script. 

3. Observe the outputs. 

#### Questions
1. How many rows and columns are in the dataset?
2. In a Pandas DataFrame, what is the index? 
3. What is the index of the 10th email message in the dataset?
4. What is the difference between the index and the contents of the 'Email no.' column?

### Correlations
The next step for data exploration is to calculate the correlations between the columns in the
dataset. The correlations may provide further insights into how words are related across emails. 

1. Run the file ```correlations.py```. This file can by run through ```sbatch```
2. If you run the file more than once, make sure you comment and uncomment the appropriate
   lines in such a way that you load the file instead of recomputing it. 
3. Observe the output. 
4. Now, change lines 56 to 60 to save the top and bottom correlations into files.

#### Questions
1. What are the top 5 correlations? 
2. What are the bottom 5 correlations?
3. What, if anything, can you infer from the top correlations? 
4. Include the lines that you modified to save the correlations in your report (number 4 above)

### Scatter Plots
Scatter plots allow us to visualize the relationships between any two variables in the dataset. 

1. Run the file ```scatter-plots.py``` and observe the output plot. 
2. You will need to modify line 36 to save the plot instead of using the show() function.

#### Questions
1. Do you observe a relationship between the words **enron** and **deal**? Explain why or why not. 
2. Create at least three more scatter plots using different word combinations. Choose the word
   combinations based on the top correlations that you found using the ```correlations.py``` file. 
3. Answer question 1 for all your scatter plots. 

### Linear regression
This file implements linear regression. 

1. Run the file ```linear-regression.py```
2. Observe the output
3. You may need to change the show() function to observe the plots. 

#### Questions
1. Why are we dropping the 'Email no' and 'Prediction' columns? 
2. What does the value 0.2 in the test_size option of the function ```train_test_split()``` mean? 
3. What is the MSE? Is this a good or bad result? Explain why. 
4. Rerun the file at least to more times with different target and input columns. Answer question 
   3 for  all  your word combinations.

### Multi-linear regression
Multi-linear regression uses the same training algorithm as linear regression but using
multiple in put variables. 

1. Run the file ```multi-linear-regression.py```
2. Observe the output. 
3. You may need to change the show() function to observe the plots. 

#### Questions
1. The coefficient output has multiple values. Why does multilinear regression have multiple
   coefficients while linear regression only has one? 
2. Modify the code to determine if there are coefficients equal (or almost) zero. Are there any
   coefficients equal to zero? If there are, what does it mean that a coefficient is zero?
3. Modify the code to use the Lasso regression algorithm. Do you get more zero coefficients?  
3. Observe the output plot carefully. You should see that some of the predictions are negative.
   However, we know that there are no negative value sin our dataset and it would make no sense
   to have a negative number of words in an email. Read this [webpage](https://scikit-learn.org/stable/modules/linear_model.html) and replace the Linear
   Regression algorithm with an appropriate algorithm for this ML task. 
4. Provide a screen shot of a plot of your results with the new algorithm that prevents negative predictions. 




## Data Reference
Balaka Biswas 
[here](https://www.kaggle.com/datasets/balaka18/email-spam-classification-dataset-csv/download?datasetVersionNumber=1). 

# logistic-regression-lab

You will complete this lab by modifying the linear regression code lab. In logistic regression, we are
interested in predicting the value of the 'Prediciton' column in the dataset. This will require that you
repeat the linear regression lab but for logistic regression. 

## Instructions

### Scatter plot
1. Copy the scatter-plots.py file and name the copy scatter-plot-classification.py
2. Modify the code to display the points on the plot according to their label in the 'Prediction' column.
   For example, the dots that correspond to spam email can be painted red while the regular email can be
   painted blue. 

#### Questions
1. Do you observe a relationship between the words **enron**, **deal**, and the dot colors? Explain why or why not. 
2. Create at least three more scatter plots using different word combinations. Choose the word
   combinations based on the top correlations that you found using the ```correlations.py``` file. 
3. Answer question 1 for all your scatter plots with colored dots. 
4. Include screen shots of your scatter plots. 

### Logistic Regression with one input variable

1. Copy the file linear-regresion.py and name the copy logistic-regression.py
2. Modify the code to ensure that the 'Prediction' column is not dropped. 
3. Modify the header to ensure that the LogisticRegression module from sickit learned is loaded
4. Modify the code that calls the LinearRegression model and instead call the LogisticRegression model. 
5. Modify the performance measure line to use an appropriate performance measure. The MSE is no longer
   appropriate in logistic regression. 
6. Remove the Coefficient and Intercept lines. 
7. The plot should show the classification results. Usually, we use a confusion matrix. Scikit learn has
   a function to create it. 

#### Questions
1. Explain what is a confusion matrix. What are the rows? What are the columns? What do you expect to see
   in the confusion matrix if your model makes perfect predictions? What if it completely wrong every
   time? 
2. Include a screen shot of your confusion matrix.

# Decision Tree Lab
In this lab, we will use a decision tree to classify the emails into spam and normal emails. 
The main advantage of decision trees is that they tell us what features are being sued to decide the
classification output. 

## Instructions
1. Run the file ```decision-tree.py```
2. Open the file ```decision-tree.png```
3. Observe the output of the .png file

### Questions
1. Read pages 176-178 in the ML book and the following:
   1. What does "enron <= 0.5" mean in the root node? 
   2. What does "sample" mean?
   3. What does the vector next to "value" mean?
   4. What does class mean?
2. What is the Gini index? 
3. What does a high/low Gini index mean for a feature? 
4. The Sikit learn decision tree depends on the following parameters:
      - max_depth: the maximum height of the decision tree. 
      - min_samples_split: the minimum number of samples a node must have before it can be split
      - min_samples_leaf: the minimum number of samples a leaf node must have
      - max_leaf_nodes: the maximum number of leaf nodes
      - max_feature: the maximum number of features that are evaluated for splitting at  each node

   Increase the max_depth of the tree and observe the confusion matrix. Did you model improve its
   predictions? Explain why or why not. 

6. Would a hiogher value for max_depth increase or decrease the capacity of the model? 
   Explain why or why not. 

7. How can we tell if the decision tree is overfitted? Briefly give an explanation. 

# Random Forest lab

## Instructions
1. Rewrite the ```decision-tree.py``` code but this time using random forests instead
   of decision trees. 
2. Compare the results of the decision tree and the random forests by comptuing their
   confusion matrices. Which one performs better? Can you improve the perfomance of the worse classifier  by changing its parameters? If so, explain. 
