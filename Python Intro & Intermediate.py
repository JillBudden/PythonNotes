#####################Intro to Python for Data Science###################
########################################################################

#####Operations with other types#####
# Several variables to experiment with
savings = 100
factor = 1.1
desc = "compound interest"

# Assign product of factor and savings to year1
year1 = factor * savings

# Print the type of year1
print(type(year1))

# Assign sum of desc and desc to doubledesc
doubledesc = desc + desc

# Print out doubledesc
print(doubledesc)


#####Type Conversion#####
# Definition of savings and result
savings = 100
result = 100 * 1.10 ** 7

# Fix the printout
print("I started with $" + str(savings) + " and now have $" + str(result) + ". Awesome!")

# Definition of pi_string
pi_string = "3.1415926"

# Convert pi_string into float: pi_float
#similar functions are str(), int(), float(), bool(), type(), len()
pi_float = float(pi_string)


#####Create a list#####
# area variables (in square meters)
hall = 11.25
kit = 18.0
liv = 20.0
bed = 10.75
bath = 9.50

# Create list areas
areas = [hall, kit, liv, bed, bath]
print(areas)


#####List of lists#####
# area variables (in square meters)
hall = 11.25
kit = 18.0
liv = 20.0
bed = 10.75
bath = 9.50

# house information as list of lists
house = [["hallway", hall],
         ["kitchen", kit],
         ["living room", liv],
         ["bedroom", bed],
         ["bathroom", bath]]

# Print out house
print(house)

# Print out the type of house
print(type(house))


#####Subsetting Lists#####
# Create the areas list
areas = ["hallway", 11.25, "kitchen", 18.0, "living room", 20.0, "bedroom", 10.75, "bathroom", 9.50]

# Print out second element from areas
print(areas[1])

# Print out last element from areas
print(areas[-1])

# Print out the area of the living room
print(areas[5])

# Create the areas list
areas = ["hallway", 11.25, "kitchen", 18.0, "living room", 20.0, "bedroom", 10.75, "bathroom", 9.50]

# Sum of kitchen and bedroom area: eat_sleep_area
eat_sleep_area = areas[3] + areas[7]
print(eat_sleep_area)

# Create the areas list
areas = ["hallway", 11.25, "kitchen", 18.0, "living room", 20.0, "bedroom", 10.75, "bathroom", 9.50]

# List slicing
# my_list[start:end] start is included while the end index is not.
# x[:2] start is 0 end is 2
# x[2:] start is 2 end is through the whole list
# x[:] is the whole list
# Use slicing to create downstairs
downstairs = areas[:6]

# Use slicing to create upstairs
upstairs = areas[6:10]

# Print out downstairs and upstairs
print(downstairs)
print(upstairs)


#####Subsetting lists of lists#####
#Row, column
x = [["a", "b", "c"],
     ["d", "e", "f"],
     ["g", "h", "i"]]

x[2][0] #returns 'g'
x[2][:2] # returns ['g', 'h']


#####Replace elements of a list#####
# Create the areas list
areas = ["hallway", 11.25, "kitchen", 18.0, "living room", 20.0, "bedroom", 10.75, "bathroom", 9.50]

# Correct the bathroom area - [-1] is the last element of a list, [-2] is the second to last element of a list, etc.
areas[-1] = 10.50

# Change "living room" to "chill zone"
areas[4] = "chill zone"


#####Extend a list#####
# Create the areas list and make some changes
areas = ["hallway", 11.25, "kitchen", 18.0, "chill zone", 20.0,
         "bedroom", 10.75, "bathroom", 10.50]

# Add poolhouse data to areas, new list is areas_1
areas_1 = areas + ["poolhouse", 24.5]

# Add garage data to areas_1, new list is areas_2
areas_2 = areas_1 + ["garage", 15.45]


#####Delete elements of a list#####
x = ["a", "b", "c", "d"]
del(x[1])


#####Copy a list#####
#changes to areas_copy do not affect the original areas list
# Create list areas
areas = [11.25, 18.0, 20.0, 10.75, 9.50]

# Create areas_copy
areas_copy = list(areas)

# Change areas_copy
areas_copy[0] = 5.0

# Print areas
print(areas)


#####Functions#####
#Help to figure out about a function, example max() or sorted()
help(max)
?max

#####String Methods#####
# string to experiment with: room
room = "poolhouse"

# Use upper() on room: room_up
room_up = room.upper()

# Print out room and room_up
print(room_up)
print(room)

# Print out the number of o's in room
print(room.count("o"))

# Create list areas
areas = [11.25, 18.0, 20.0, 10.75, 9.50]

# Print out the index of the element 20.0
print(areas.index(20.0))

# Print out how often 14.5 appears in areas
print(areas.count(14.5))

# Create list areas
areas = [11.25, 18.0, 20.0, 10.75, 9.50]

# Use append twice to add poolhouse and garage size
areas.append(24.5)
areas.append(15.45)

# Print out areas
print(areas)

# Reverse the orders of the elements in areas
areas.reverse()

# Print out areas
print(areas)

#Other list methods: append(), remove(), reverse()


#####Numpy#####
# Create list baseball
baseball = [180, 215, 210, 210, 188, 176, 209, 200]

# Import the numpy package as np
import numpy as np

# Create a Numpy array from baseball: np_baseball
np_baseball = np.array(baseball)

# Print out type of np_baseball
print(type(np_baseball))


#################Intermediate Python for Data Science###################
########################################################################

#####matplotlib#####

#####plotline#####
import matplotlib.pyplot as plt
plt.plot(x,y)
plt.show()


######scatter#####
import matplotlib.pyplot as plt
plt.scatter(x,y)
plt.xscale('log') #scale
xlab = 'GDP per Capita [in USD]'
ylab = 'Life Expectancy [in years]'
title = 'World Development in 2007'
plt.xlabel(xlab) #labels
plt.ylabel(ylab)
plt.title(title) #title
tick_val = [1000,10000,100000] #tick values
tick_lab = ['1k','10k','100k'] #tick lables
plt.xticks(tick_val, tick_lab) #adapt the ticks on the x-axis
plt.show() #show the plot

#Additional Customizations of a scatter plot
# Scatter plot
plt.scatter(x = gdp_cap, y = life_exp, s = np.array(pop) * 2, c = col, alpha = 0.8)

# Previous customizations
plt.xscale('log') 
plt.xlabel('GDP per Capita [in USD]')
plt.ylabel('Life Expectancy [in years]')
plt.title('World Development in 2007')
plt.xticks([1000,10000,100000], ['1k','10k','100k'])

# Additional customizations
plt.text(1550, 71, 'India')
plt.text(5700, 80, 'China')

# Add grid() call
plt.grid(True)

# Show the plot
plt.show()


######histogram, with 5 bins#####
import matplotlib.pyplot as plt
plt.hist(life_exp, bins=5)
plt.show()
plt.clf() #clean up plot


#####Dictionaires#####
# Definition of countries and capital
countries = ['spain', 'france', 'germany', 'norway']
capitals = ['madrid', 'paris', 'berlin', 'oslo']

# From string in countries and capitals, create dictionary europe. Key:Value pairs
europe = {
    'spain': 'madrid',
    'france': 'paris',
    'germany': 'berlin',
    'norway': 'oslo',
}

# Print europe
print(europe)

# Definition of dictionary
europe = {'spain':'madrid', 'france':'paris', 'germany':'berlin', 'norway':'oslo' }

# Print out the keys in europe
print(europe.keys())

# Print out value that belongs to key 'norway'
print(europe['norway'])


#####Add new Key:Value pair#####
# Definition of dictionary
europe = {'spain':'madrid', 'france':'paris', 'germany':'berlin', 'norway':'oslo' }

# Add italy to europe
europe['italy'] = 'rome'

# Print out italy in europe
print('italy' in europe)

# Add poland to europe
europe['poland'] = 'warsaw'

# Print europe
print(europe)


#####Delete a Key:Value pair#####
# Definition of dictionary
europe = {'spain':'madrid', 'france':'paris', 'germany':'bonn',
          'norway':'oslo', 'italy':'rome', 'poland':'warsaw',
          'australia':'vienna' }

# Update capital of germany
europe['germany'] = 'berlin'

# Remove australia
del(europe['australia'])

# Print europe
print(europe)


#####Dictionary of dictionaires#####
# Dictionary of dictionaries
europe = { 'spain': { 'capital':'madrid', 'population':46.77 },
           'france': { 'capital':'paris', 'population':66.03 },
           'germany': { 'capital':'berlin', 'population':80.62 },
           'norway': { 'capital':'oslo', 'population':5.084 } }


# Print out the capital of France
print(europe['france']['capital'])

# Create sub-dictionary data
data = {'capital': 'rome', 'population': 59.83}

# Add data to europe under key 'italy'
europe['italy'] = data

# Print europe
print(europe)


####Dictionary to Dataframe#####
# Import pandas
import pandas as pd

# Build cars DataFrame
names = ['United States', 'Australia', 'Japan', 'India', 'Russia', 'Morocco', 'Egypt']
dr =  [True, False, False, False, True, True, True]
cpc = [809, 731, 588, 18, 200, 70, 45]
dict = { 'country':names, 'drives_right':dr, 'cars_per_cap':cpc }
cars = pd.DataFrame(dict)
print(cars)

# Definition of row_labels
row_labels = ['US', 'AUS', 'JAP', 'IN', 'RU', 'MOR', 'EG']

# Specify row labels of cars
cars.index = row_labels

# Print cars again
print(cars)


#####CSV to dataframe#####
# Import pandas as pd
import pandas as pd

# Fix import by including index_col - so the first column is used as ROW labels
cars = pd.read_csv('cars.csv', index_col = 0)

# Print out country column as Pandas Series
print(cars['country']) #single bracket gives a Pandas Series

# Print out country column as Pandas DataFrame
print(cars[['country']]) #double bracket gives a Pandas DataFrame

# Print out DataFrame with country and drives_right columns
print(cars[['country', 'drives_right']])

# Print out first 3 observations - first 3 rows
print(cars[0:3])

# Print out fourth, fifth and sixth observation - rows
print(cars[3:6])

# Print out cars
print(cars)

#loc and iloc use for data selection an doperation on DataFrames
#loc is label-based, which means you have to specify rows and columns based on row and column labels
#iloc is integer index based, need to specify rows and colmns by their integer index
#loc and iloc allow you to select both rows and columns from a dataframe
#each pair of commands gives the same result:
cars.loc['RU']
cars.iloc[4]

cars.loc[['RU']]
cars.iloc[[4]]

cars.loc[['RU', 'AUS']]
cars.iloc[[4, 1]]

cars.loc['IN', 'cars_per_cap']
cars.iloc[3, 0]

cars.loc[['IN', 'RU'], 'cars_per_cap']
cars.iloc[[3, 4], 0]

cars.loc[['IN', 'RU'], ['cars_per_cap', 'country']]
cars.iloc[[3, 4], [0, 1]]

#it is possible to select onlycolumns with loc and iloc
# ":" equals the whole row or column
cars.loc[:, 'country']
cars.iloc[:, 1]

cars.loc[:, ['country','drives_right']]
cars.iloc[:, [1, 2]]

#example

# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Print out drives_right column as Series
x = cars.loc[:, 'drives_right']
print(x)

# Print out drives_right column as DataFrame
print(cars.loc[:, ['drives_right']])

# Print out cars_per_cap and drives_right as DataFrame
z = cars.loc[:, ['cars_per_cap', 'drives_right']]
print(z)


#####Control Flow#####
#example 1
area = 10.0

if(area < 9) :
    print("small")
elif(area < 12) :
    print("medium")
else :
    print("large")

#example 2
# Define variables
room = "bed"
area = 14.0

# if-elif-else construct for room
if room == "kit" :
    print("looking around in the kitchen.")
elif room == "bed":
    print("looking around in the bedroom.")
else :
    print("looking around elsewhere.")

# if-elif-else construct for area
if area > 15 :
    print("big place!")
elif area > 10:
    print("medium size, nice!")
else :
    print("pretty small.")


#####Filtering a Pandas DataFrame#####
#example1
# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Extract drives_right column as Series: dr
dr = cars['drives_right']

# Use dr to subset cars: sel
sel = cars[dr]

# Print sel
print(sel)

#example 2
# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Import numpy, you'll need this
import numpy as np

# Create medium: observations with cars_per_cap between 100 and 500
cpc = cars['cars_per_cap']
between = np.logical_and(cpc > 100, cpc < 500)
medium = cars[between]

# Print medium
print(medium)


#####Loops#####

#####While loop#####

# Initialize offset
offset = 8

# Code the while loop
while offset != 0:
    print("correcting...")
    offset = offset - 1
    print(offset)


#####While loop with conditionals#####

# Initialize offset
offset = -6

# Code the while loop
while offset != 0 :
    print("correcting...")
    if offset > 0:
        offset = offset - 1
    else:
        offset = offset + 1
    print(offset)


#####Loop over a list#####
# areas list
areas = [11.25, 18.0, 20.0, 10.75, 9.50]

# Code the for loop
for x in areas:
    print(x)


#####Enumerate#####
#If you also want to access the index information
# areas list
areas = [11.25, 18.0, 20.0, 10.75, 9.50]

# Change for loop to use enumerate()
for index, x in enumerate(areas):
    print("room " + str(index) + ": " + str(x))


#####Loop over a list of lists#####
# house list of lists
house = [["hallway", 11.25], 
         ["kitchen", 18.0], 
         ["living room", 20.0], 
         ["bedroom", 10.75], 
         ["bathroom", 9.50]]
         
# Build a for loop from scratch
for x in house:
    print("the " + str(x[0]) + " is " + str(x[1]) + " sqm")


#####Loop over a dictionary#####
#in Python 3 you need the items() method to loop over a dictionary

# Definition of dictionary
europe = {'spain':'madrid', 'france':'paris', 'germany':'bonn', 
          'norway':'oslo', 'italy':'rome', 'poland':'warsaw', 'australia':'vienna' }
          
# Iterate over europe
for key, value, in europe.items():
    print("the capital of " + key + " is " + value)


#####Loop over a DataFrame#####
# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Iterate over rows of cars
for lab, row in cars.iterrows() :
    print(lab)
    print(row)

# Adapt for loop
for lab, row in cars.iterrows() :
    print(lab + ": " + str(row['cars_per_cap']))

# Code for loop that adds COUNTRY column
for lab, row in cars.iterrows():
    cars.loc[lab, "COUNTRY"] = row["country"].upper()

# Print cars
print(cars)

#other example with loop creating a column
# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Use .apply(str.upper)
for lab, row in cars.iterrows() :
    cars["COUNTRY"] = cars["country"].apply(str.upper)

print(cars)
