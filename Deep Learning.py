####DEEP LEARNING
#Example as seen by linear regression: interactions
#Neural networks account for interactions really well
#Build models using keras
#Models account for interactions with seen variables (input & output layers); "hidden layer" are called nodes - similar to latent variables
#Forward propagation: multiply-add process; these weights are put in as weights ex: 'node_1': np.array([1, 2])
#NumPy is implicit in all as "np" in all examples


#####Forward Propagation - Neural Network#####
#Input data is pre-loaded as "input_data"
#Weights are available in a dictionare called "weights"
#The array of weights for the first node in the hidden layer are in weights['node_0']
#The array of weights for the second node in the hidden layer are in weights['node_1']
#The model will predict how many transactions the user makes in the next year.

# Calculate node 0 value: node_0_value
# The multiply-add process 
node_0_value = (input_data * weights['node_0']).sum()

# Calculate node 1 value: node_1_value
# The multiply-add process
node_1_value = (input_data * weights['node_1']).sum()

# Put node values into array: hidden_layer_outputs
hidden_layer_outputs = np.array([node_0_value, node_1_value])

# Calculate output: output
# generate the prediction by multiplying "hidden_layer_outputs" by "weights['output']" and computing their sum
output = (hidden_layer_outputs * weights['output']).sum()

# Print output
print(output)

##The network generated a prediction of -39; see the next section to correct this!


#####Using Activation Functions#####
#For neural networks to achieve their greatest predictive power need Activation Functions
#Activation functions allow the model to caputre nonlinearities
#Activation functions are applied to node inputs to produce node output
#The standard activation function to use is the ReLU (Rectified Linear Activation)
#ReLU is essentially two linear functions put together
#ReLU function takes a single number as an input, returning 0 if the input is negative, and the input if the input is positive

def relu(input):
    '''Define your relu activation function here'''
    # Calculate the value for the output of the relu function: output
    output = max(0, input)
    
    # Return the value just calculated
    return(output)

# Calculate node 0 value: node_0_output
# Additional step added from the first example code
# Apply the relu() function to node_0_input to calculate node_0_output
node_0_input = (input_data * weights['node_0']).sum()
node_0_output = relu(node_0_input)

# Calculate node 1 value: node_1_output
# Additional step added from the first example code
# Apply the relu() function to node_1_input to calculate node_1_output
node_1_input = (input_data * weights['node_1']).sum()
node_1_output = relu(node_1_input)

# Put node values into array: hidden_layer_outputs
hidden_layer_outputs = np.array([node_0_output, node_1_output])

# Calculate model output (do not apply relu)
model_output = (hidden_layer_outputs * weights['output']).sum()

# Print model output
print(model_output)

#output is 52 transactions predicted, correcting the previous negative prediction, next need to tune model weights


#####Applying the Network to Many Observations/Rows of Data#####
#define a function to generate predictions for multiple data observations pre-loaded as "input_data"
#"weights" are also pre-loaded
#relu() function from previous example is pre-loaded

# Define the function predict_with_network() that accepts two arguments: "input_data_row" and "weights" 
# returns a prediction fromt the network as the output 
def predict_with_network(input_data_row, weights):

    # Calculate node 0 value
    node_0_input = (input_data_row * weights['node_0']).sum()
    node_0_output = relu(node_0_input)

    # Calculate node 1 value
    node_1_input = (input_data_row * weights['node_1']).sum()
    node_1_output = relu(node_1_input)

    # Put node values into array: hidden_layer_outputs
    hidden_layer_outputs = np.array([node_0_output, node_1_output])
    
    # Calculate model output
    input_to_final_layer = (hidden_layer_outputs * weights['output']).sum()
    model_output = relu(input_to_final_layer)
    
    # Return model output
    return(model_output)

# Create empty list to store prediction results
# Use a for loop to iterate over "input_data"
results = []
for input_data_row in input_data:
    # Append prediction to results
	# Use "predict_with_network()" function to generate preictions for each row of the "input_data" - "input_data_row"
	# .append each prediction to "results"
    results.append(predict_with_network(input_data_row, weights))

# Print results
print(results)
        
# output generates predictions for each


#####Multiple Hidden Layers - Multi-layer Neural Networks#####
#Representation learning: Deep Networks internally build representations of patterns in the data
#partially replace the need for feature engineering
#subsequent layers build increasingly sophisticated representations of raw data

#Deep Learning: modeler doesn't need to specify the interactions
#When you train the model, the neural network gets weights that find the relevant patterns to make better predictions

#2 Hidden layers
#pre-loaded: input_data
#pre-loaded first hidden layer: node_0_0, node_0_1, weights['node_0_0'], weights['node_0_1']
#pre-loaded second hidden layer: node_1_0 and node_1_1; weights['node_1_0'] and weights['node_1_1'] 
#pre-loaded: weights['output']

def predict_with_network(input_data):
    # Calculate node 0 in the first hidden layer
	# Calculate node_0_0_input using its weights weights['node_0_0'] and the given input_data
	# Then apply the relu function to get node_0_0_output
    node_0_0_input = (input_data * weights['node_0_0']).sum()
    node_0_0_output = relu(node_0_0_input)

    # Calculate node 1 in the first hidden layer
    node_0_1_input = (input_data * weights['node_0_1']).sum()
    node_0_1_output = relu(node_0_1_input)

    hidden_0_outputs = np.array([node_0_0_output, node_0_1_output])

    # Calculate node 0 in the second hidden layer
    node_1_0_input = (hidden_0_outputs * weights['node_1_0']).sum()
    node_1_0_output = relu(node_1_0_input)

    # Calculate node 1 in the second hidden layer
    node_1_1_input = (hidden_0_outputs * weights['node_1_1']).sum()
    node_1_1_output = relu(node_1_1_input)

    hidden_1_outputs = np.array([node_1_0_output, node_1_1_output])
    
    # Calculate output here: model_output
	# Calculate model_output using its weights weights['output'] and the outputs from the second hidden layer hidden_1_outputs array
	# Do not apply the relu function to this output.
    model_output = (hidden_1_outputs * weights['output']).sum()
    
    # Return model_output
    return(model_output)

output = predict_with_network(input_data)
print(output)

#The network genrated a prediction of 364 


####Need for optimization#####
#EX: When the multiple-add process results in the output node of "9", but the actual value of the target is 13.
#Error = actual - predicted (13 - 9)
#Changing the weights changes the predictions to decrease the error
#Loss function: Mean Squared Error
#Lower loss function value means a better model (i.e., less error)
#Goal: to find the weights that give the lowest value for the loss function: Gradient Descent
#Gradient descent: start at a random point until you are somewhere flat... find the slope and take a step downhill


####Coding how weight changes affect accuracy#####

# The data point you will make a prediction for
input_data = np.array([0, 3])

# Sample weights
weights_0 = {'node_0': [2, 1],
             'node_1': [1, 2],
             'output': [1, 1]
            }

# The actual target value, used to calculate the error
target_actual = 3

# Make prediction using original weights
# use predict_with_network() function
model_output_0 = predict_with_network(input_data, weights_0)

# Calculate error: error_0
error_0 = model_output_0 - target_actual

# Create weights that cause the network to make perfect prediction (3): weights_1
# Here ONE weight from weights_0 has been changed****
weights_1 = {'node_0': [2, 1],
             'node_1': [1, 2],
             'output': [1, 0]
            }

# Make prediction using new weights: model_output_1
# use predict_with_network() function
model_output_1 = predict_with_network(input_data, weights_1)

# Calculate error: error_1
# subtract target_actual from model_output_1 
error_1 = model_output_1 - target_actual

# Print error_0 and error_1
print(error_0)
print(error_1)

#results indicated the first model had an error of 6, and the updated model has an error of 0 - a perfect prediction


#####Scaling up to multiple data points#####
#Write code to compare model accuracies for two different sets of weights: weights_0 and weights_1
#Other implicit in the code: input_data is a list of arrays, each item in the list contains data to make a single prediction
#Target_actuals is a list of numbers; each item in teh list is the actual value we are trying to predict
#Use the mean_squared_error() function from sklearn.metrics - takes true values and predicted values as arguments
#Use predict_with_networ() function, which takes an array of data as the first argument, and weights as the second argument

from sklearn.metrics import mean_squared_error

# Create model_output_0 
model_output_0 = []
# Create model_output_0
model_output_1 = []

# for Loop over input_data
for row in input_data:
    # Append prediction to model_output_0
	# Make predictions for each row with weights_0 using the predict_with_network() function and append it to model_output_0
    model_output_0.append(predict_with_network(row, weights_0))
    
    # Append prediction to model_output_1
    model_output_1.append(predict_with_network(row, weights_1))

# Calculate the mean squared error for model_output_0: mse_0
# Calculate the mean squared error of model_output_0 and then model_output_1 
# using the mean_squared_error() function. The first argument should be the predicted values, 
# and the second argument should be the true values (target_actuals).
mse_0 = mean_squared_error(model_output_0, target_actuals)

# Calculate the mean squared error for model_output_1: mse_1
mse_1 = mean_squared_error(model_output_1, target_actuals)

# Print mse_0 and mse_1
print("Mean squared error with weights_0: %f" %mse_0)
print("Mean squared error with weights_1: %f" %mse_1)

#Output shows the differences in Mean Squared Error between the 2 models - you want lower error!


####Gradient Descent - calculate slopes and update weights####
#Gradient descent: update each weight by subtracting learning rate * slope
#Slope calculation
#To calculate the slope for a weight, need to multiply: 
# 1. slope of the loss function w/ respect to value at the node we feed into
# 2. the vlue of the node that feeds into our weight
# 3. Slope of the activation function w/ respect to value we feed into (if you have an activation function)
# example: if the learning rate is 0.01 and the slope of the mean square loss function is -24, and the current weight is 2:
# updated the new weight to: 2 - 0.01(-24) = 2.24

####Calculating Slopes####
#When plotting the mean-squared error loss function against predictions, the slope is 2 * x * (y-xb), or 2 * input_data * error. 
#Note that x and b may have multiple numbers (x is a vector for each data point, and b is a vector).
#Code to calculate this slope while using a single data point. 
#You'll use pre-defined weights called weights as well as data for a single point called input_data. 
#The actual value of the target you want to predict is stored in target

# Calculate the predictions: preds
preds = (weights * input_data).sum()

# Calculate the error: error
error = target - preds

# Calculate the slope: slope
slope = 2 * input_data * error

# Print the slope
print(slope)

#Use the printed slope(s) in improve the weights of the model!


#####Improving Model Weights#####
#you've calculated teh slopes you need in the last example, now use those slopes to improve the model
#the weights have been pre-loaded as "weights",
#the actual value of the target as "target",
#and the input data as "input_data"
#The predictions from the initial weights are stored as preds

# Set the learning rate: learning_rate
# Set it to be 0.01
learning_rate = 0.01

# Calculate the predictions: preds
preds = (weights * input_data).sum()

# Calculate the error: error
error = target - preds

# Calculate the slope: slope
slope = 2 * input_data * error

# Update the weights: weights_updated
weights_updated = weights + learning_rate * slope

# Get updated predictions: preds_updated
preds_updated = (weights_updated * input_data).sum()

# Calculate updated error: error_updated
error_updated = target - preds_updated

# Print the original error
print(error)

# Print the updated error
print(error_updated)

#results showed that compared to the earlier model, updating the model weights decreased the error


####Making Multiple Updates to Weights#####
#pre-loaded: get_slope(), input_data, target, weights, met_mse(), matplotlib.pyplot
#This model does not have any hidden layers, this is why weights is a single array

'''Using a for loop to iteratively update weights:
Calculate the slope using the get_slope() function.
Update the weights using a learning rate of 0.01.
Calculate the mean squared error (mse) with the updated weights using the get_mse() function.
Append mse to mse_hist.'''

n_updates = 20
mse_hist = []

# Iterate over the number of updates
for i in range(n_updates):
    # Calculate the slope: slope
    slope = get_slope(input_data, target, weights)
    
    # Update the weights: weights
    weights = weights + 0.01 * slope
    
    # Calculate mse with new weights: mse
    mse = get_mse(input_data, target, weights)
    
    # Append the error to mse_hist
    mse_hist.append(mse)

# Plot the error history
plt.plot(mse_hist)
plt.xlabel('Iterations')
plt.ylabel('Mean Squared Error')
plt.show()

#Result shows the mean squared error decreases as the number of iterations go up


####Backpropagation####
#Skipping this for now


###########################Creating a Keras Model#########################
#####Specifying a Model#####
'''Now you'll get to work with your first model in Keras, and will immediately be able to run more 
complex neural network models on larger datasets compared to the first two chapters.
To start, you'll take the skeleton of a neural network and add a hidden layer and an output layer. 
You'll then fit that model and see Keras do the optimization so your model continually gets better.
As a start, you'll predict workers wages based on characteristics like their industry, education and 
level of experience. You can find the dataset in a pandas dataframe called df. For convenience, 
everything in df except for the target has been converted to a NumPy matrix called predictors. 
The target, wage_per_hour, is available as a NumPy matrix called target. For all exercises in this 
chapter, we've imported the Sequential model constructor, the Dense layer constructor, and pandas'''

# Import necessary modules
import keras
from keras.layers import Dense
from keras.models import Sequential

# Save the number of columns in predictors: n_cols
n_cols = predictors.shape[1]

# Set up the model: model
# a Sequential model called model 
model = Sequential()

# Add the first layer
# Use the .add() method on model to add a Dense layer
# Add 50 units, specify activation='relu', and the input_shape parameter to be the tuple (n_cols,)
# which means it has n_cols items in each row of data, and any number of rows of data are accepatble as inputs
model.add(Dense(50, activation='relu', input_shape=(n_cols,)))

# Add the second layer
model.add(Dense(32, activation='relu'))

# Add the output layer
model.add(Dense(1))


####Compiling the model####
#now that a model has been specified, the next step is to compile it
#Specify the optimizer - controls the learning rate - "Adam" is usually a good choice
#Specify the Loss function
#Mean Squared Error is common for regression
#Fitting a model: Applying Backpropation and gradient descent with your data to update the weights
#Scaling data before optimization is important

#To complie the model you need to specify the optimizer an dloss function to use****

# Import necessary modules
import keras
from keras.layers import Dense
from keras.models import Sequential

# Specify the model
n_cols = predictors.shape[1]
model = Sequential()
model.add(Dense(50, activation='relu', input_shape = (n_cols,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# Compile the model****
model.compile(optimizer='adam', loss='mean_squared_error')

# Verify that model contains information from compiling
print("Loss function: " + model.loss)


#####Fit the Model#####
'''You're at the most fun part. You'll now fit the model. Recall that the data to be 
used as predictive features is loaded in a NumPy matrix called predictors and the data 
to be predicted is stored in a NumPy matrix called target. Your model is pre-written and 
it has been compiled with the code from the previous exercise.'''

# Import necessary modules
import keras
from keras.layers import Dense
from keras.models import Sequential

# Specify the model
n_cols = predictors.shape[1]
model = Sequential()
model.add(Dense(50, activation='relu', input_shape = (n_cols,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Fit the model
# Fit the "model", first arguement is the "predictors", and the data to be predicted ("target") is the second
model.fit(predictors, target)


#####Classification Models#####
#use the loss function of 'categorical_crossentropy' loss function, not mean squared error
#output layer has separate node for each possible outcome (i.e., 2 columns instead of 1)
#Uses 'softmax' activation function
#consider the Kaggle Titantic dataset for practice here - example code is applied to this dataset
#The titantic dataset is pre-loaed into a DataFrame called df
#predictive variables are in a NumPy array 'predictors', and the target to predict is df.survived
#The number of predictive features is stored in n_cols
#use the 'sgd' optimizer (Stochastic Gradient Descent)

# Import necessary modules
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical

# Convert the target to categorical: target
# Convert df.survived to a categorical variable using the to_categorical() function 
target = to_categorical(df.survived)

# Set up the model
# specify a Sequential model called model 
model = Sequential()

# Add the first layer
# Add a Dense layer with 32 nodes. Use 'relu' as the activation and (n_cols,) as the input_shape
model.add(Dense(32, activation='relu', input_shape=(n_cols,)))

# Add the output layer
# Add the Dense output layer. Because there are two outcomes, it should have 2 units, 
# and because it is a classification model, the activation should be 'softmax'
model.add(Dense(2, activation='softmax'))

# Compile the model
# Compile the model, using 'sgd' as the optimizer, 'categorical_crossentropy' as the loss function, 
# and metrics=['accuracy'] to see the accuracy (what fraction of predictions were correct) at 
# the end of each epoch
model.compile(optimizer='sgd', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Fit the model using the predictors and the target 
model.fit(predictors, target)

#Results indicate this simple model is generating an accuracy of 68.


#####Using Models#####
#save the model, reload it, make predictions, verify it's structure


#####Making Predictions#####
'''The trained network from your previous coding exercise is now stored as model. 
New data to make predictions is stored in a NumPy array as pred_data. Use model 
to make predictions on your new data. In this exercise, your predictions will be 
probabilities, which is the most common way for data scientists to communicate 
their predictions to colleagues.'''

# Specify, compile, and fit the model
model = Sequential()
model.add(Dense(32, activation='relu', input_shape = (n_cols,)))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='sgd', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])
model.fit(predictors, target)

# Calculate predictions: predictions
# create your predictions using the model's .predict() method on pred_data
predictions = model.predict(pred_data)

# Calculate predicted probability of survival: predicted_prob_true
'''Use NumPy indexing to find the column corresponding to predicted 
probabilities of survival being True. This is the second column (index 1) 
of predictions. Store the result in predicted_prob_true and print it.'''
predicted_prob_true = predictions[:,1]

# print predicted_prob_true
print(predicted_prob_true)

#Output prints all of the predictions!


####Optimizing/fine-tuning keras models####
#learning rate to low or too high, or just right?
#This example will utilize low, high, and 'just right' learning rates
#A low value for a loss function is good
'''You'll want the optimization to start from scratch every time you change 
the learning rate, to give a fair comparison of how each learning rate did 
in your results. So we have created a function get_new_model() that creates 
an unoptimized model to optimize'''

# Import the SGD optimizer
from keras.optimizers import SGD

# Create list of learning rates to try optimizing with: lr_to_test
lr_to_test = [.000001, 0.01, 1]

# Loop over learning rates
for lr in lr_to_test:
    print('\n\nTesting model with learning rate: %f\n'%lr )
    
    # Build new model to test, unaffected by previous models
	# Use the get_new_model() function to build a new, unoptimized model
    model = get_new_model()
    
    # Create SGD optimizer with specified learning rate: my_optimizer
    my_optimizer = SGD(lr=lr)
    
    # Compile the model
	# Set the optimizer parameter to be the SGD object you created above
	# because this is a classificaton problem, use 'categorical_crossentropy' for the loss parameter 
    model.compile(optimizer=my_optimizer, loss='categorical_crossentropy')
    
    # Fit the model
    model.fit(predictors, target)
	

####Model Validation - Evaluating model accuracy on validation dataset####
#Use a hold out sample - or a validation split
#Early stopping - use it to stop optimization when it isn't helping anymore.
#Model definition provided as "model"

'''Compile your model using 'adam' as the optimizer and 'categorical_crossentropy' 
for the loss. To see what fraction of predictions are correct (the accuracy) in each 
epoch, specify the additional keyword argument metrics=['accuracy'] in model.compile().
Fit the model using the predictors and target. Create a validation split of 30% (or 0.3). 
This will be reported in each epoch.'''

# Save the number of columns in predictors: n_cols
n_cols = predictors.shape[1]
input_shape = (n_cols,)

# Specify the model
model = Sequential()
model.add(Dense(100, activation='relu', input_shape = input_shape))
model.add(Dense(100, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit the model
hist = model.fit(predictors, target, validation_split=0.3)


#####Now Add Early Stopping#####
#Since the optimization stops automatically when it isn't helping, 
# you can set a high valuye for epochs in your call to .fit()
#the model you'll optimize has been spcefied as model#data is pre-loaded as preictors and target

# Import EarlyStopping
from keras.callbacks import EarlyStopping

# Save the number of columns in predictors: n_cols
n_cols = predictors.shape[1]
input_shape = (n_cols,)

# Specify the model
model = Sequential()
model.add(Dense(100, activation='relu', input_shape = input_shape))
model.add(Dense(100, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compile the model
# Compile the model, once again using 'adam' as the optimizer, 'categorical_crossentropy' 
# as the loss function, and metrics=['accuracy'] to see the accuracy at each epoch.
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define early_stopping_monitor - stop optimaization when the validation loss hasn't improved for 2 epochs 
early_stopping_monitor = EarlyStopping(patience=2)

# Fit the model - specify 30 epochs (default is 10... but because of early stopping 30 is okay) and a validation split of 0.3
# pass [early_stopping_monitor] to the callbacks parameter
model.fit(predictors, target, epochs=30, validation_split=0.3, callbacks=[early_stopping_monitor])

#Results showed that the optimization stopped after 7 epochs 


#####Experimenting with Wider Networks#####
#Begin experimenting with different models!
#a model called "model_1" is pre-loaded, it is small and has 10 units in each hidden layer
#Here you create a new model called "model_2", except it has 100 units in each hidden layer
'''After you create model_2, both models will be fitted, and a graph showing both models loss 
score at each epoch will be shown. We added the argument verbose=False in the fitting commands 
to print out fewer updates, since you will look at these graphically instead of as text.'''

# Define early_stopping_monitor
early_stopping_monitor = EarlyStopping(patience=2)

# Create the new model: model_2
'''Create model_2 to replicate model_1, but use 100 nodes instead of 10 for the 
first two Dense layers you add with the 'relu' activation. Use 2 nodes for the 
Dense output layer with 'softmax' as the activation.'''
model_2 = Sequential()

# Add the first and second layers
model_2.add(Dense(100, activation='relu', input_shape=input_shape))
model_2.add(Dense(100, activation='relu'))

# Add the output layer
model_2.add(Dense(2, activation='softmax'))

# Compile model_2
'''Compile model_2 as you have done with previous models: Using 'adam' as the optimizer, 
'categorical_crossentropy' for the loss, and metrics=['accuracy']'''
model_2.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])

# Fit model_1
model_1_training = model_1.fit(predictors, target, epochs=15, validation_split=0.2, callbacks=[early_stopping_monitor], verbose=False)

# Fit model_2
model_2_training = model_2.fit(predictors, target, epochs=15, validation_split=0.2, callbacks=[early_stopping_monitor], verbose=False)

# Create the plot
plt.plot(model_1_training.history['val_loss'], 'r', model_2_training.history['val_loss'], 'b')
plt.xlabel('Epochs')
plt.ylabel('Validation score')
plt.show()

#Results are graphical rather than text - the blue model is the one you made
#Red is the original model
#The blue model had a lower loss value, so it is the better model


#####Adding Hidden Layers to a Network#####
#The previous example was how you experiment with wider networks, now try deeper networks
'''Once again, you have a baseline model called model_1 as a starting point. It has 1 hidden 
layer, with 50 units. You can see a summary of that model's structure printed out. You will 
create a similar network with 3 hidden layers (still keeping 50 units in each layer).'''

'''pecify a model called model_2 that is like model_1, but which has 3 hidden layers of 50 
units instead of only 1 hidden layer.
Use input_shape to specify the input shape in the first hidden layer.
Use 'relu' activation for the 3 hidden layers and 'softmax' for the output 
layer, which should have 2 units. Compile model_2 as you have done with previous 
models: Using 'adam' as the optimizer, 'categorical_crossentropy' for the loss, 
and metrics=['accuracy']. Hit 'Submit Answer' to fit both the models and visualize 
which one gives better results! For both models, you should look for the best val_loss 
and val_acc, which won't be the last epoch for that model.'''

# The input shape to use in the first hidden layer
input_shape = (n_cols,)

# Create the new model: model_2
model_2 = Sequential()

# Add the first, second, and third hidden layers
model_2.add(Dense(50, activation='relu', input_shape=input_shape))
model_2.add(Dense(50, activation='relu'))
model_2.add(Dense(50, activation='relu'))

# Add the output layer
model_2.add(Dense(2, activation='softmax'))

# Compile model_2
model_2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit model 1
model_1_training = model_1.fit(predictors, target, epochs=20, validation_split=0.4, callbacks=[early_stopping_monitor], verbose=False)

# Fit model 2
model_2_training = model_2.fit(predictors, target, epochs=20, validation_split=0.4, callbacks=[early_stopping_monitor], verbose=False)

# Create the plot
plt.plot(model_1_training.history['val_loss'], 'r', model_2_training.history['val_loss'], 'b')
plt.xlabel('Epochs')
plt.ylabel('Validation score')
plt.show()

#Red is the original model
#The blue model had a lower loss value, so it is the better model


####Thinking about Model Capacity (or network capacity)#####
#Similar to overfitting and underfitting 
#Overfitting: the ability of a model to fit aspects of the data that are happenstance - it won't generalize to new data 
#(will make inaccurate predictions in validation data) - so accurate in training data, inaccurate in validation data set
# increasing layers in your model moves you further toward overfitting 
#Underfitting: opposite. Model fails to find patterns in trainin data (nor validation data) - inaccurate in both training and validation datasets
#Model Capacity is the ability of your model to capture predictive patterns in your data 

#Workflow for optimizing model capacity:
#1. start with a small network
#2. get the validation score
#3. keep increasing capacity until validation score is no longer improving
#Ex: conduct sequential experiments: hidden layers, nodes, mean squared error... next step increase or decrease capacity 


####Stepping Up to Images - Building Your Own Digit Recognition Model####
#MNIST dataset - recognizing handwritten digits
#X and y loaded and ready to model with. Sequential and Dense from keras are also pre-imported
'''If you have a computer with a CUDA compatible GPU, you can take advantage of it to improve computation time. 
If you don't have a GPU, no problem! You can set up a deep learning environment in the cloud that can run your 
models on a GPU'''
#see here to do cloud computing: https://www.datacamp.com/community/tutorials/deep-learning-jupyter-aws#gs.D1DwGDc
#or need NVIDIA GPU with MacBook

# Create the model: model
model = Sequential()

# Add the first hidden layer
# Add the first Dense hidden layer of 50 units to your model with 'relu' activation. For this data, the input_shape is (784,)
model.add(Dense(50, input_shape=(784,), activation='relu'))

# Add the second hidden layer
model.add(Dense(50, activation='relu'))

# Add the output layer
# Activation function is softmax
# The number of nodes in this layer should be the same as the number of possible outputs, in this case 10
model.add(Dense(10, activation='softmax'))

# Compile the model
# Compile using adam as the optimizer
# categorical_crossentropy for the loss 
# and metrics=['accuracy'] 
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#Fit the model
# use a valication split of 0.3
model.fit(X, y, validation_split=0.3)

#Model shows 90% accuracy in recognizing handwirtten digits.