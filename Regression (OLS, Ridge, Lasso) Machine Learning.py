#Supervised learning with regression
#Target variable is continuous
#fit a line to the data
#minimize the sum of the squares of residuals (Ordinary Least Squares: OLS)
# y = ax + b y = target, x = single feature, a and b are parameters
# R2 is percent of variance explained

#############################################################################################################
#####Preprocessing data#####
#Scikit-learn will not accept non-numerical features by default
#For example a "region" variable containing different regions will need to be dummy coded inorder for it to be used as a predictor
#Convert to dummy variables: 0 & 1
#Create dummy variables: scikit-learn OneHotEncoder()
#Create dummy variables: pandas get_dummies()

#####Explore categorical features#####
# Import pandas
import pandas as pd

# Read 'gapminder.csv' into a DataFrame: df
df = pd.read_csv('gapminder.csv')

# Create a boxplot of life expectancy per region
df.boxplot('life', 'Region', rot=60)

# Show the plot
plt.show()


#####Create dummy variables#####
# Create dummy variables: df_region
df_region = pd.get_dummies(df)

# Print the columns of df_region
print(df_region.columns)

# Drop 'Region_America' (the unneeded dummy variable) from df_region
df_region = pd.get_dummies(df, drop_first=True)

# Print the new columns of df_region
print(df_region.columns)


#####Handling Missing data#####
#educated guess
#impute the mean (mean replace)
#delete
#scikit-learn pipeline 
#if missing data is NaN can use pandas methods .dropna() and .fillna() or scikit-learn Imputer()

#####convert '?' to NaN and then drop the rows that contain them from the DataFrame#####

# Convert '?' to NaN
df[df == '?'] = np.nan

# Print the number of NaNs
print(df.isnull().sum())

# Print shape of original DataFrame
print("Shape of Original DataFrame: {}".format(df.shape))

# Drop missing values and print shape of new DataFrame
# Drops all rows with missing data 
df = df.dropna()

# Print shape of new DataFrame
print("Shape of DataFrame After Dropping All Rows with Missing Values: {}".format(df.shape))


#####Imputing Missing data in a ML Pipeline#####
#SVC = Support Vector Classification, a type of SVM 
#Utlizes the Support Vector Machine or SVM

# Import the Imputer module
from sklearn.preprocessing import Imputer
from sklearn.svm import SVC

# Setup the Imputation transformer: imp
# Setup the Imputation transformer to impute missing data 
# represented as 'NaN') with the 'most_frequent' value in the column (axis=0).
imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)

# Instantiate the SVC classifier: clf
clf = SVC()

# Setup the pipeline with the required steps: steps
# The first tuple should consist of the imputation step, using imp.
# The second should consist of the classifier.
steps = [('imputation', imp),
        ('SVM', clf)]


#####Imputing missing data in a ML pipeline II#####
#illusration only, note this uses K-NN or logistic regression data (i.e., classification)
# Import necessary modules
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# Setup the pipeline steps: steps
steps = [('imputation', Imputer(missing_values='NaN', strategy='most_frequent', axis=0)),
        ('SVM', SVC())]
        
# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit the pipeline to the train set
pipeline.fit(X_train, y_train)

# Predict the labels of the test set
y_pred = pipeline.predict(X_test)

# Compute metrics
print(classification_report(y_test, y_pred))


#####Centering and Scaling Data#####
#For data that is on different scales, standardization/mean centering is very important!
#Featurs on larger scales can unduly influence the model
#We want features to be on a similar scale
#mean = 0, sd = 1
#Subtract the minimum and divide by the range (minimum zero and maximum 1)
#Normalize the data to ranges from -1 to +1

# Import scale
from sklearn.preprocessing import scale

# Scale the features X using scale(): X_scaled
X_scaled = scale(X)

# Print the mean and standard deviation of the unscaled features
print("Mean of Unscaled Features: {}".format(np.mean(X))) 
print("Standard Deviation of Unscaled Features: {}".format(np.std(X)))

# Print the mean and standard deviation of the scaled features
print("Mean of Scaled Features: {}".format(np.mean(X_scaled))) 
print("Standard Deviation of Scaled Features: {}".format(np.std(X_scaled)))



####################################################################################################
#####Importing the data and getting into the right shape#####
# Import numpy and pandas
import numpy as np
import pandas as pd

# Read the CSV file into a DataFrame: df
df = pd.read_csv('gapminder.csv')

# Create arrays for features and target variable
y = df['life'].values
X = df['fertility'].values

# Print the dimensions of X and y before reshaping
print("Dimensions of y before reshaping: {}".format(y.shape))
print("Dimensions of X before reshaping: {}".format(X.shape))

# Reshape X and y
y = y.reshape(-1, 1)
X = X.reshape(-1, 1)

# Print the dimensions of X and y after reshaping
print("Dimensions of y after reshaping: {}".format(y.shape))
print("Dimensions of X after reshaping: {}".format(X.shape))


#####EDA#####
#Utlize .info() .describe(), .head()
#Seaborn heatmap using df.corr() which computes  the pairwise correlations between columns:
sns.heatmap(df.corr(), square=True, cmap='RdYlGn')


#####Fit & Predict#####
#with one predictor
#the array for the target variable is implicit in the below code as y and the array for features is X_fertility

# Import LinearRegression
from sklearn.linear_model import LinearRegression

# Create the regressor: reg
reg = LinearRegression()

# Create the prediction space to range from the min to the max of X_fertility
prediction_space = np.linspace(min(X_fertility), max(X_fertility)).reshape(-1,1)

# Fit the model to the data (fit the regressor to the data)
reg.fit(X_fertility, y)

# Compute predictions over the prediction space: y_pred
# (compute predictions using the .predict() method and the prediction_space array)
y_pred = reg.predict(prediction_space)

# Print R^2 using .score() method
print(reg.score(X_fertility, y))

# Plot regression line
# (overlay the plot with your linear regression line)
plt.plot(prediction_space, y_pred, color='black', linewidth=3)
plt.show()


#####Train/test split#####
#to help ensure results generalize to new data
#fit and predict liear regression over ALL features
#In addition to computing R2, will also compute Root Mean Squared Error (RMSE)
#feature array X and target variable array y are implicit in the below code

# Import necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Create training and test sets
# 30% is used for testing and 70% for training. Use a random state of 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

# Create the regressor: reg_all
reg_all = LinearRegression()

# Fit the regressor to the training data
reg_all.fit(X_train, y_train)

# Predict on the test data: y_pred
y_pred = reg_all.predict(X_test)

# Compute and print R^2 and RMSE
print("R^2: {}".format(reg_all.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))


#####Cross-validation#####
#essentially, splitting up datafile and running analyses 
#It maximizes the amount of data that is used to train the model
#k-folds CV (ex: 5-fold CV... split data into 5 samples) - the function will return 5 scores; you need to take the average of these
#cross_val_score() uses R2 as the metric of choice for regression\
#X, y are implict in the below code, as well as various necessary import statements

# Import the necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# Create a linear regression object: reg
reg = LinearRegression()

# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(reg, X, y, cv=5)

# Print the 5-fold cross-validation scores
print(cv_scores)

# Print the average 5-fold cross-validation score using NumPy's mean() function
print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))


#####Explore different CV folds#####
#use the following code to compare time it takes for k-folds: %timeit cross_val_score(reg, X, y, cv = ____)


#####Regularized Regression#####
#Linear regression minimizes a loss function (minimizes the sum of the squared residuals)
#It chooses a coefficent for each feature variable and large coefficients can lead to overfitting
#To correct this regularize, which penalizes large coefficients
#example 1: Ridge regression
#Alpha is a parameter we need to choose (hyperparameter tuning)
#Alpha controls model complexity
#When Alpha=0 we get back OLS
#very high alpha can lead to underfitting
#Example 2: Lasso regression 
#Lasso regression can be used to select important features of a dataset
#It shrinks the coefficients of less important features to exactly 0
#the power of reporting important variables is important in real world settings

#####Regularization I: Lasso#####
#identify the most important feature, less important features are shrunk to 0
#The feature and target variable arrays are implicit in the code as X and y_pred

# Import lasso
from sklearn.linear_model import Lasso

# Instantiate a lasso regressor: lasso
# alpha of 0.4 and specify normalize=True 
lasso = Lasso(alpha=0.4, normalize=True)

# Fit the regressor to the data 
lasso.fit(X, y)

# Compute and print the coefficients using the coef_ attribute
lasso_coef = lasso.coef_
print(lasso_coef)

# Plot the coefficients
# In the plot note which features are 0 and which is not 
plt.plot(range(len(df_columns)), lasso_coef)
plt.xticks(range(len(df_columns)), df_columns.values, rotation=60)
plt.margins(0.02)
plt.show()


#####Regularization II: Ridge#####
#Lasso is great for feature selection, but Ridge regression should be your first choice
#In this example fit ridge regression models over a range of different alphas and plot CV R2 for each
''' def display_plot(cv_scores, cv_scores_std):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alpha_space, cv_scores)

    std_error = cv_scores_std / np.sqrt(10)

    ax.fill_between(alpha_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2)
    ax.set_ylabel('CV Score +/- Std Error')
    ax.set_xlabel('Alpha')
    ax.axhline(np.max(cv_scores), linestyle='--', color='.5')
    ax.set_xlim([alpha_space[0], alpha_space[-1]])
    ax.set_xscale('log')
    plt.show() '''

# Import necessary modules
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# Setup the array of alphas and lists to store scores
alpha_space = np.logspace(-4, 0, 50)
ridge_scores = []
ridge_scores_std = []

# Create a ridge regressor: ridge
ridge = Ridge(normalize=True)

# Compute scores over range of alphas
for alpha in alpha_space:

    # Specify the alpha value to use: ridge.alpha
    ridge.alpha = alpha
    
    # Perform 10-fold CV: ridge_cv_scores
	# Data is available in the arrays X and y 
    ridge_cv_scores = cross_val_score(ridge, X, y, cv=10)
	
	# Append the mean of ridge_cv_scores to ridge_scores
	# Numpy has been pre-limported as np 
    ridge_scores.append(np.mean(ridge_cv_scores))
    
    # Append the std of ridge_cv_scores to ridge_scores_std
    ridge_scores_std.append(np.std(ridge_cv_scores))

# Display the plot
display_plot(ridge_scores, ridge_scores_std)


#####Hold-out set#####
#For Lasso and ridge regression Lasso uses the L1 penalty to regularize, while ridge uses the L2 penalty.
#Another type of regularized regression is the elastic net.
#Elastic net is a linear combination of the L1 and L2 penalties.
#In scikit-learn this is represented by the 'l1_ratio' parameter: An 'l1_ratio' of 1 corresponds to an L1L1 penalty, 
#and anything lower is a combination of L1L1 and L2L2

# Import necessary modules
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=42)

# Create the hyperparameter grid
# Specify the hyperparameter grid for 'l1_ratio' using l1_space as the grid of values to search over.
l1_space = np.linspace(0, 1, 30)
param_grid = {'l1_ratio': l1_space}

# Instantiate the ElasticNet regressor: elastic_net
elastic_net = ElasticNet()

# Setup the GridSearchCV object with 5-fold CV: gm_cv
gm_cv = GridSearchCV(elastic_net, param_grid, cv=5)

# Fit it to the training data
gm_cv.fit(X_train, y_train)

# Predict on the test set and compute metrics
y_pred = gm_cv.predict(X_test)
r2 = gm_cv.score(X_test, y_test)
mse = mean_squared_error(y_test, y_pred)
print("Tuned ElasticNet l1 ratio: {}".format(gm_cv.best_params_))
print("Tuned ElasticNet R squared: {}".format(r2))
print("Tuned ElasticNet MSE: {}".format(mse))


#####Bring it all together: Pipeline for regression#####
#build a pipeline that imputes missing data, scales features, and fits an ElasticNet using GridSearchCV
#All necessary modules are implicit, as well as X and y

# Setup the pipeline steps: steps
# 'imputation', which uses the Imputer() transformer and the 'mean' strategy to impute missing data ('NaN') using the mean of the column.
# 'scaler', which scales the features using StandardScaler().
# 'elasticnet', which instantiates an ElasticNet regressor.
steps = [('imputation', Imputer(missing_values='NaN', strategy='mean', axis=0)),
         ('scaler', StandardScaler()),
         ('elasticnet', ElasticNet())]
         
# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Specify the hyperparameter space for the l1 ratio
# sing the following notation: 'step_name__parameter_name'. 
# Here, the step_name is elasticnet, and the parameter_name is l1_ratio 
parameters = {'elasticnet__l1_ratio':np.linspace(0,1,30)}

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Create the GridSearchCV object; uses 3-fold CV which is default, so doesn't need to be specified: gm_cv
gm_cv = GridSearchCV(pipeline, parameters)

# Fit the GridSearchCV object to the training set
gm_cv.fit(X_train, y_train)

# Compute and print the metrics
r2 = gm_cv.score(X_test, y_test)
print("Tuned ElasticNet Alpha: {}".format(gm_cv.best_params_))
print("Tuned ElasticNet R squared: {}".format(r2))
