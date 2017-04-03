#Train & evaluate supervised learning models
#supervised learning models are labeled data in contrast to unsupervised
#k-Nearest Neighbors is for classification
#Data example is congressional voting records - predicting party membership based on voting record
#UCI Machine Learning Repo: https://archive.ics.uci.edu/ml/datasets/Congressional+Voting+Records
#Data Features need to be in an array
#Feature array = x (predictor/independent variables) and response variablee array = y (outcome/dependent variable)


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


