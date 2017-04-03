#Train & evaluate supervised learning models
#supervised learning models are labeled data in contrast to unsupervised
#k-Nearest Neighbors is for classification
#Data example is congressional voting records - predicting party membership based on voting record
#UCI Machine Learning Repo: https://archive.ics.uci.edu/ml/datasets/Congressional+Voting+Records
#Data Features need to be in an array
#Feature array = x (predictor/independent variables) and response variablee array = y (outcome/dependent variable)



####See Regression Machine Learning for preprocessing data#####
#missing data
#creating dummy variables, if needed
#Centering & Scaling



#####Numerical EDA#####
#utilize .head() .info() .describe()


#####Visual EDA#####
#Utilize Seaborn 
#Example EDA code:
plt.figure()
sns.countplot(x='education', hue='party', data=df, palette='RdBu')
plt.xticks([0,1], ['No', 'Yes'])
plt.show()


#####Fit#####
#Fit a k-Nearest Neightbors classifier to the voting dataset
#The target needs to be a single column with the same number of observations as the feature data.

# Import KNeighborsClassifier from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier 

# Create arrays for the features and the response variable
# Note .drop() to drop the target variable 'party' from teh feature array X
# .values ensures X and Y are NumPy arrays. Without using .values, X and Y are a DataFrame and Series, in this form scikit-learn API will accept them as long as they are in the right shape
y = df['party'].values
X = df.drop('party', axis=1).values

# Create a k-NN classifier with 6 neighbors
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the data
knn.fit(X, y)


#####Predict#####
#Having fit a k-NN classifier, you can not use it to predict the label of a new data point.
#However, there is no unlabeled data since all was used to fit the model!
#Can still use the .predict() method on the X that was used to fit the model, but it's not a good indicator of the model's ability to generalize to new, unseen data.
#For now, a random unlabled data point had been generated: X_new : use your classifier to predict the label for this new data point
#note, using it on X will generate 435 predictions

# Import KNeighborsClassifier from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier 

# Create arrays for the features and the response variable
y = df['party'].values
X = df.drop('party', axis=1).values

# Create a k-NN classifier with 6 neighbors: knn
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the data
knn.fit(X, y)

# Predict the labels for the training data X: y_pred
y_pred = knn.predict(X)
print("Prediction: {}".format(y_pred))

# Predict and print the label for the new data point X_new
new_prediction = knn.predict(X_new)
print("Prediction: {}".format(new_prediction)) 


####Evaluate the model#####
#Accuracy
#training and test sets
#compare predictions with labeled data
#Model complexity: as larger K = less complex model and can lead to underfit model; smaller K = more complex model and can lead to overfitting


#####Example with more than a binary outcome and text/train sets#####
#Uses teh MNIST data 0-9 categorical outcomes

# Import necessary modules
from sklearn import datasets
import matplotlib.pyplot as plt

# Load the digits dataset: digits
digits = datasets.load_digits()

# Print the keys and DESCR of the dataset
print(digits.keys())
print(digits.DESCR)

# Print the shape of the images and data keys
print(digits.images.shape)
print(digits.data.shape)

# Display the 1010th image using plt.imshow()
plt.imshow(digits.images[1010], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()

# Import necessary modules
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split

# Create feature and target arrays
X = digits.data
y = digits.target

# Split into training and test set
# use 0.2 for the size of the test set and use a random state of 42
# stratify the split according to the labels so that they are distributed in the training and test sets as they are in the original dataset 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y)

# Create a k-NN classifier with 7 neighbors: knn
knn = KNeighborsClassifier(n_neighbors=7)

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Print the accuracy of the classifier's predictions using the .score() method
print(knn.score(X_test, y_test))


#####Overfitting & Underfitting#####
#Model complexity curve
#Compute and plot the training and testing accuracy scores for a variety of different neighbor values
#Basically, looking at accuracy scores and how they differ with different values of k and which is underfitting and overfitting
#X_train, X_test, y_train, Y_test are implicit in the below code as well as other necessary import statements

# Setup arrays to store train and test accuracies
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over different values of k
for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit the classifier to the training data
    knn.fit(X_train, y_train)
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)

    #Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_test)

# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()

#####Model Evaluation for Classification modelds#####
#Accuracy is not always good
#Confusion matrix: true positive, flase negative, false positive, true negative
#Metrics from the confusion matrix = Precision, recall, F1 score
#Example, high precision: not many real emails predicted as span
#Example, high recall: predicted most spam emails correctly
#note: support gives the number of samples of the true response that lie in that class
#data from UCI ML repo: https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes
#X and y and some necessary imports are implicit in the code
# sklearn.model_selection.train_test_split and sklearn.neighbors.KNeighborsClassifier

# Import necessary modules
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Create training and test set, with 40% of the data used for testing. Use a random state of 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=42)

# Instantiate a k-NN classifier: knn
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Predict the labels of the test data: y_pred
y_pred = knn.predict(X_test)

# Generate the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


#####Logistic regression & the ROC curve#####
#logistic regression creates a linear decision boundary
#threshold = 0.50
#Vary the threshold from 1 - 0 = the ROC curve
#fpr = false positive rate
#tpr = true positive rate

#Feature and target variable arrays X and y are implicit 
#train_test_split has been imported for you from sklearn.model_selection

# Import the necessary modules
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# Create training and test sets with 40% of the data used for testing, with a random state of 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=42)

# Create the classifier: logreg
# (Instantiate a LogisticRegression classifier called logreg)
logreg = LogisticRegression()

# Fit the classifier to the training data
logreg.fit(X_train, y_train)

# Predict the labels of the test set: y_pred
y_pred = logreg.predict(X_test)

# Compute and print the confusion matrix and classification report
# See how logistic regression compares to k-NN! /// logistic regression wins 
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


#####Plotting a ROC curve#####
#utlize .predict_proba()

# Import necessary modules
from sklearn.metrics import roc_curve

# Using logreg classifier, which has been fit to the training data
# Compute predicted probabilities of the labels of the test set X_test: y_pred_prob
y_pred_prob = logreg.predict_proba(X_test)[:,1]

# Generate ROC curve values: fpr, tpr, thresholds
# fpr = false positive rate
# tpr = true positive rate
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Plot ROC curve
# The larger the area under the ROC curve = better model
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


#####Precision-recall curve#####
#plot the precision and recall for different thresholds
#precision= tp / tp + fp
#recall = tp / tp + fn


#####AUC computation#####
#Area Under ROC Curve
#say you have a binary classifier that is just randomly making guesses. 
#It would be correct 50% of the time and the ROC curve would be a diagonal line 
#The True Positive Rate and teh False Positive Rate would be equal and the area under the ROC curve would be 0.5
#Hence if the Area Under the ROC curve (AUC) is greater than 0.5, the model is better than random guessing.
#calculate AUC scores using roc_auc_score() from sklearn.metrics 
#X_train, X_test, y_train, y_test and logistic regression classifier logreg is implicit

# Import necessary modules
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score

# Using the logreg classifier, which has beeen fit to the training data, 
# Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg.predict_proba(X_test)[:,1]

# Compute the AUC score using the roc_auc_score() function, the test set labels y_test, and the predicted probabilities y_pred_prob
print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))

# Compute cross-validated AUC scores: cv_auc
cv_auc = cross_val_score(logreg, X, y, cv=5, scoring='roc_auc')

# Print list of AUC scores
print("AUC scores computed using 5-fold cross-validation: {}".format(cv_auc))


#####Hyperparameter tuning#####
#Try a bunch of different hyperparameter values
#Fit all of them separately
#See how well each performs
#Choose the best performing one
#Essential to use cross-validation 
#Grid search cross-validation
#Logistic regression has a regularization parameter: C
#C controls the inverse of the regulatarization strength; a large C can lead to an overfit model, a small C to an underfit model

#use GridSearchCV and logistic regression to find the optimal C in a hyperparameter space

# Import necessary modules
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Setup the hyperparameter grid
# setup the hyperparameter grid by using c_space as the grid of values to tune C over 
c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space}

# Instantiate a logistic regression classifier: logreg
logreg = LogisticRegression()

# Instantiate the GridSearchCV object with 5-fold CV: logreg_cv
logreg_cv = GridSearchCV(logreg, param_grid, cv=5)

# Fit it to the data
# use the .fit() method on the GridSearchCV object to fit it to the data X and y 
logreg_cv.fit(X, y)

# Print the tuned parameter and score
print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_))
print("Best score is {}".format(logreg_cv.best_score_))

#Output will show which value of C results in the best performance


#####Hyperparameter tuning with RandomizedSearchCV#####
#GridSearchCV can be computationally expensive. A Solution is to use RandomizedSearchCV where NOT all hyperparameter values are tried out
#A fixed number of hyperparameter settings are sampled from probability distributions
#Note randomizedSearchCV will never outperform GridSearchCV. instead it is valuable because it sames on computation time.
#Will use a Decision Tree

# Import necessary modules
from scipy.stats import randint
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV

# Setup the parameters and distributions to sample from: param_dist
param_dist = {"max_depth": [3, None],
              "max_features": randint(1, 9),
              "min_samples_leaf": randint(1, 9),
              "criterion": ["gini", "entropy"]}

# Instantiate a Decision Tree classifier: tree
tree = DecisionTreeClassifier()

# Instantiate the RandomizedSearchCV object with 5-fold CV: tree_cv
# Insde RandomizedSearchCV() specify the classifier, parameter distribution, and number of folds to use.
tree_cv = RandomizedSearchCV(tree, param_dist, cv=5)

# Fit it to the data
# Use the .fit() method on the RandomizedSearchCV object to fit it to the data X and y 
tree_cv.fit(X, y)
 
# Print the tuned parameters and score
# Print the best parameter and best score
print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
print("Best score is {}".format(tree_cv.best_score_))


#####Hold-out set#####
#evaluate a model with tuned hyperparameters on a hold-out set
#In addition to C, logistic regression has a 'penalty' hyperparameter which specifieds whether to use 'l1' or 'l2' regularization
#so, create a hold-out set, tune the 'C' and 'penalty' hyperparameters using GridSearchCV  on the training set
#then evaluate its performance againt the hold-out set

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Create the hyperparameter grid
# use the array c_space as the grid of values for 'C'
# For penalty, specify a list consting of l1 abd l2 
c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space, 'penalty': ['l1', 'l2']}

# Instantiate the logistic regression classifier: logreg
logreg = LogisticRegression()

# Create train and test sets
# In practice the test set functions as the hold-out 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=42)

# Instantiate the GridSearchCV object using 5-folds cv: logreg_cv
logreg_cv = GridSearchCV(logreg, param_grid, cv=5)

# Fit it to the training data
logreg_cv.fit(X_train, y_train)

# Print the optimal parameters and best score
print("Tuned Logistic Regression Parameter: {}".format(logreg_cv.best_params_))
print("Tuned Logistic Regression Accuracy: {}".format(logreg_cv.best_score_))


#####Bring it all together: Pipeline for classification#####
#Scaling and hyperparameter tuning 
#The hyperparameters you will tune are C and gamma. C controls the regularization strength
#it is analogous to the C used for logistic regresion
#gamma controls the kernel coefficient.
#The following are implicit (preloaded): Pipeline, svm, train_test_split, GridSearchCV, Classification_report, accuracy_score, X, y 

# Setup the pipeline
# Scaling, called 'scaler' with StandardScaler().
# Classification, called 'SVM' with SVC()
steps = [('scaler', StandardScaler()),
         ('SVM', SVC())]

pipeline = Pipeline(steps)

# Specify the hyperparameter space
# 'step_name__parameter_name'. Here, the step_name is SVM, and the parameter_names are C and gamma.
parameters = {'SVM__C':[1, 10, 100],
              'SVM__gamma':[0.1, 0.01]}

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

# Instantiate the GridSearchCV object; 3-fold CV is default so don't have to specify it: cv
cv = GridSearchCV(pipeline, parameters)

# Fit to the training set
cv.fit(X_train, y_train)

# Predict the labels of the test set: y_pred
y_pred = cv.predict(X_test)

# Compute and print metrics
print("Accuracy: {}".format(cv.score(X_test, y_test)))
print(classification_report(y_test, y_pred))
print("Tuned Model Parameters: {}".format(cv.best_params_))






