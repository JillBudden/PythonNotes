#####Random Float#####
#seed() sets the random seed, so results are reproducible between simulations
#rand() if you don't specify any arguements, generates a random float between 0 and 1

# Import numpy as np
import numpy as np

# Set the seed
np.random.seed(123)

# Generate and print random float
x = np.random.rand()
print(x)


#####Roll the dice#####
#use randint() to generate integers randomly

# Import numpy and set seed
import numpy as np
np.random.seed(123)

# Use randint() to simulate a dice
x = np.random.randint(1, 7)
print(x)

# Use randint() again
y = np.random.randint(1, 7)
print(y)


####Stepping and the Empire State Building#####
# Import numpy and set seed
import numpy as np
np.random.seed(123)

# Starting step
step = 50

# Roll the dice
dice = np.random.randint(1, 7)

# Finish the control construct
if dice <= 2 :
    step = step - 1
elif dice <= 5:
    step = step + 1
else :
    step = step + np.random.randint(1,7)

# Print out dice and step
print(dice)
print(step)


#####Use a For Loop to similate a random walk#####
#includes visulizing the walk

# Import numpy and set seed
import numpy as np
np.random.seed(123)

# Initialize random_walk
random_walk = [0]

# Complete 
for x in range(100) :
    # Set step: last element in random_walk
    step = random_walk[-1]

    # Roll the dice
    dice = np.random.randint(1,7)

    # Determine next step
    if dice <= 2:
        # use max to make sure step can't go below 0
        step = max(0, step - 1)
    elif dice <= 5:
        step = step + 1
    else:
        step = step + np.random.randint(1,7)

    # append next_step to random_walk
    random_walk.append(step)

# Print random_walk
print(random_walk)

# Import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# Plot random_walk
plt.plot(random_walk)


#####Simulate multiple walks#####
#to get an idea about how your chances are of reaching 60 steps, repeatedly simulate the random walk

# Initialization
import numpy as np
import matplotlib.pyplot as plt

# Set the seed
np.random.seed(123)

# Initialize all_walks
all_walks = []

# Simulate random walk 10 times
for i in range(10) :

    # Code from the initial random walk (see above)
    random_walk = [0]
    for x in range(100) :
        step = random_walk[-1]
        dice = np.random.randint(1,7)

        if dice <= 2:
            step = max(0, step - 1)
        elif dice <= 5:
            step = step + 1
        else:
            step = step + np.random.randint(1,7)
        random_walk.append(step)

		# Implement clumsiness -  you have a 0.1% chance of falling down
        if np.random.rand(0, 1) <= 0.001 :
            step = 0

    # Append random_walk to all_walks
    all_walks.append(random_walk)

# Print all_walks
print(all_walks)

# Visualize all walks
# Convert all_walks to Numpy array: np_aw
np_aw = np.array(all_walks)

# Plot np_aw and show
plt.plot(np_aw)
plt.show()

# Clear the figure
plt.clf()

# Transpose np_aw: np_aw_t
np_aw_t = np.transpose(np_aw)

# Plot np_aw_t and show
plt.plot(np_aw_t)
plt.show()