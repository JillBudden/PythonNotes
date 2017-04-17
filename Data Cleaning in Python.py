####Loading & Viewing your Data####

# Import pandas
import pandas as pd

# Read the file into a DataFrame: df
df = pd.read_csv('dob_job_application_filings_subset.csv')

# Print the head of df
print(df.head())

# Print the tail of df
print(df.tail())

# Print the shape of df
print(df.shape)

# Print the columns of df
print(df.columns)

# Print the head and tail of df_subset
print(df_subset.head())
print(df_subset.tail())

# Print the info of df
print(df.info())

# Print the info of df_subset
print(df_subset.info())


####Summary Stats####
#.describe() // can only be used on numeric columns otherwise use .value_counts() (see below)


####Frequency Counts for Categorical Data####
# Print the value counts for 'Borough'
print(df['Borough'].value_counts(dropna=False))

# Print the value_counts for 'State'
print(df['State'].value_counts(dropna=False))

# Print the value counts for 'Site Fill'
print(df['Site Fill'].value_counts(dropna=False))


####Visualizing single variables with histograms####

# Import matplotlib.pyplot
import matplotlib.pyplot as plt

# Plot the histogram
df['Existing Zoning Sqft'].plot(kind='hist', rot=70, logx=True, logy=True)

# Display the histogram
plt.show()


####Visualize Multiple Variables with Boxplots####
# Import necessary modules
import pandas as pd
import matplotlib.pyplot as plt

# Create the boxplot
df.boxplot(column='initial_cost', by='Borough', rot=90)

# Display the plot
plt.show()


####Multiple Variables with Scatter Plots####
# Import necessary modules
import pandas as pd
import matplotlib.pyplot as plt

# Create and display the first scatter plot
df.plot(kind='scatter', x='initial_cost', y='total_est_fee', rot=70)
plt.show()

# Create and display the second scatter plot
df_subset.plot(kind='scatter', x='initial_cost', y='total_est_fee', rot=70)
plt.show()


####Tidy Data: Reshaping data using Melt####
#For data to be tidy each variable is a separate column and each row is a separate observation
#Meltng is the process of turning columns of data into rows of data
#Melting takes a set of columns and turns it into a single column

# Print the head of airquality
print(airquality.head())

# Melt airquality: airquality_melt
airquality_melt = pd.melt(airquality, id_vars=['Month', 'Day'])

# Print the head of airquality_melt
print(airquality_melt.head())


####Customizing Melted Data####
#renaming

# Print the head of airquality
print(airquality.head())

# Melt airquality: airquality_melt
airquality_melt = pd.melt(airquality, id_vars=['Month', 'Day'], var_name='measurement', value_name='reading')

# Print the head of airquality_melt
print(airquality_melt.head())


####Pivoting Data####
#Pivoting data is the opposite of melting
#Pivoting will create a new column for each unique value in a specified column
#.pivot_table() has an index parameter which you can use to specify the columns that you don't want pivoted

# Print the head of airquality_melt
print(airquality_melt.head())

# Pivot airquality_melt: airquality_pivot
airquality_pivot = airquality_melt.pivot_table(index=['Month', 'Day'], columns='measurement', values='reading')

# Print the head of airquality_pivot
print(airquality_pivot.head())


####Resetting the index of a DataFrame####
#After pivoting you get back a pandas dataframe with a hierarchical index (also known as a multi-index)
#Need to use .reset_index()

# Print the index of airquality_pivot
print(airquality_pivot.index)

# Reset the index of airquality_pivot: airquality_pivot
airquality_pivot = airquality_pivot.reset_index()

# Print the new index of airquality_pivot
print(airquality_pivot.index)

# Print the head of airquality_pivot
print(airquality_pivot.head())


####Pivoting Duplicate Values####
#By using the .pivot_table() and the aggfunc parameter, you can not only reshape your data, but also remove duplicates
#Followed by flatten the columns of the pivoted DataFrame using .reset_index()

# Pivot airquality_dup: airquality_pivot
airquality_pivot = airquality_dup.pivot_table(index=['Month', 'Day'], columns='measurement', values='reading', aggfunc=np.mean)

# Reset the index of airquality_pivot
airquality_pivot = airquality_pivot.reset_index()

# Print the head of airquality_pivot
print(airquality_pivot.head())

# Print the head of airquality
print(airquality.head())


####Data Cleaning: Splitting a column with a .str####
#ex: gender and age in one column, but want it in 2 columns ("M014")
#use string indexing to split the data 

# Melt tb: tb_melt
tb_melt = pd.melt(tb, id_vars=['country', 'year'])

# Create the 'gender' column
tb_melt['gender'] = tb_melt.variable.str[0]

# Create the 'age_group' column
tb_melt['age_group'] = tb_melt.variable.str[1:]

# Print the head of tb_melt
print(tb_melt.head())


####Splitting a column with .split() and .get()####
#multiple variables may be stored in columns with a delimiter
#ex: '_' may serve as a delimiter 

# Melt ebola: ebola_melt
ebola_melt = pd.melt(ebola, id_vars=['Date', 'Day'], var_name='type_country', value_name='counts')

# Create the 'str_split' column
'''Create a column called 'str_split' by splitting ebola_melt.type_country on '_'. 
Note that you will first have to access the str attribute of type_country before 
you can use split().'''
ebola_melt['str_split'] = ebola_melt.type_country.str.split('_')

# Create the 'type' column
'''Create a column called 'type' by using the .get() method to retrieve 
index 0 of ebola_melt.str_split.'''
ebola_melt['type'] = ebola_melt.str_split.str.get(0)

# Create the 'country' column
'''Create a column called 'country' by using the .get() 
method to retrieve index 1 of ebola_melt.str_split'''
ebola_melt['country'] = ebola_melt.str_split.str.get(1)

# Print the head of ebola_melt
print(ebola_melt.head())


####Concatenating Data####
#Combining rows of data
#Similar to combining 3 datasets together, 1 added on to the bottom of another

# Concatenate uber1, uber2, and uber3: row_concat
row_concat = pd.concat([uber1, uber2, uber3])

# Print the shape of row_concat
print(row_concat.shape)

# Print the head of row_concat
print(row_concat.head())


####Combining Columns of Data####
#Stiching data together fromt he sides instead of the top and bottom

# Concatenate ebola_melt and status_country column-wise: ebola_tidy
ebola_tidy = pd.concat([ebola_melt, status_country], axis=1)

# Print the shape of ebola_tidy
print(ebola_tidy.shape)

# Print the head of ebola_tidy
print(ebola_tidy.head())


####Merging Data 1-to-1####
#merge on a common, unique variable

# Merge the DataFrames: o2o
'''Merge the site and visited DataFrames on the 'name' 
column of site and 'site' column of visited'''
o2o = pd.merge(left=site, right=visited, left_on='name', right_on='site')

# Print o2o
print(o2o)


####Many-to-1 Data Merge####
#Here one of the values will be duplicated and recycled in the output (one of the keys is not unqiue)

# Merge the DataFrames: m2o
'''Merge the site and visited DataFrames on the 'name' column of site and 'site' column of visited, 
exactly as you did in the previous exercise'''
m2o = pd.merge(left=site, right=visited, left_on='name', right_on='site')

# Print m2o
print(m2o)


####Many-to-Many Data Merge####
#When both datasets do not have a unique key for a merge
#What happens here is that for each duplicated key, every pairwise combination will be created

# Merge site and visited: m2m
'''Merge the site and visited DataFrames on the 'name' column of site and 'site' column of visited, 
exactly as you did in the previous two exercises. Save the result as m2m'''
m2m = pd.merge(left=site, right=visited, left_on='name', right_on='site')

# Merge m2m and survey: m2m
'''Merge the m2m and survey DataFrames on the 'ident' column of m2m and 'taken' column of survey.'''
m2m = pd.merge(left=m2m, right=survey, left_on='ident', right_on='taken')

# Print the first 20 lines of m2m
print(m2m.head(20))


####Converting Data Types####
#Ensure all categorical variables in the DataFrame are of type "category" - saves on memoryview

# Convert the sex column to type 'category'
tips.sex = tips.sex.astype('category')

# Convert the smoker column to type 'category'
tips.smoker = tips.smoker.astype('category')

# Print the info of tips
print(tips.info())


####Working with Numeric Data####
#numeric data is int or float. 
#If, instead it is type "object", it usually means there is non numeric value in the column
#Use pd.to_numeric() function to convert a column to a numeric data type
#"coerce" a value into a missing value ("NaN")

# Convert 'total_bill' to a numeric dtype
# Coerce the errors to NaN by specifying the keyword argument errors='coerce'.
tips['total_bill'] = pd.to_numeric(tips['total_bill'], errors='coerce')

# Convert 'tip' to a numeric dtype
tips['tip'] = pd.to_numeric(tips['tip'], errors='coerce')

# Print the info of tips
print(tips.info())


####String parsing with regular expressions####
#Regular Expressions: https://docs.python.org/3/library/re.html
'''The regular expression module in python is re. When performing pattern matching 
on data, since the pattern will be used for a match across multiple rows, it's 
better to compile the pattern first using re.compile(), and then use the compiled 
pattern to match values.'''

# Import the regular expression module
import re

# Compile the pattern: prog
'''Compile a pattern that matches a phone number of the format xxx-xxx-xxxx.
Use \d{x} to match x digits. Here you'll need to use it three times: twice 
to match 3 digits, and once to match 4 digits.
Place the regular expression inside re.compile().'''
prog = re.compile('\d{3}-\d{3}-\d{4}')

# See if the pattern matches
'''Using the .match() method on prog, check whether the pattern 
matches the string '123-456-7890'.'''
result = prog.match('123-456-7890')
print(bool(result))

# See if the pattern matches
result = prog.match('1123-456-7890')
print(bool(result))


####Extracting Numerical Values From Strings####
'''When using a regular expression to extract multiple numbers (or multiple 
pattern matches, to be exact), you can use the re.findall() function. Dan 
did not discuss this in the video, but it is straightforward to use: You pass 
in a pattern and a string to re.findall(), and it will return a list of the matches.'''

# Import the regular expression module
import re

# Find the numeric values: matches
'''Write a pattern that will find all the numbers in the following string: 'the recipe 
calls for 10 strawberries and 1 banana'. To do this:
Use the re.findall() function and pass it two arguments: the pattern, followed by the string.
\d is the pattern required to find digits. This should be followed with a + so that 
the previous element is matched one or more times. This ensures that 10 is viewed as one 
number and not as 1 and 0.'''
matches = re.findall('\d+', 'the recipe calls for 10 strawberries and 1 banana')

# Print the matches
print(matches)


####Pattern Matching####
#Using Regular Expressions

'''Write patterns to match:
A telephone number of the format xxx-xxx-xxxx. You already did this in a previous exercise.
A string of the format: A dollar sign, an arbitrary number of digits, a decimal point, 2 digits.
Use \$ to match the dollar sign, \d* to match an arbitrary number of digits, \. to match the 
decimal point, and \d{x} to match x number of digits.
A capital letter, followed by an arbitrary number of alphanumeric characters.
Use [A-Z] to match any capital letter followed by \w* to match an arbitrary number of 
alphanumeric characters.
'''

# Write the first pattern
pattern1 = bool(re.match(pattern='\d{3}-\d{3}-\d{4}', string='123-456-7890'))
print(pattern1)

# Write the second pattern
pattern2 = bool(re.match(pattern='\$\d*\.\d{2}', string='$123.45'))
print(pattern2)

# Write the third pattern
pattern3 = bool(re.match(pattern='[A-Z]\w*', string='Australia'))
print(pattern3)


####Custom Fuctions to Clean Data####
#Recoding variables
#Use the .apply() method to apply a function across entire rows or columns of DataFrames

# Define recode_sex()
def recode_sex(sex_value):

    # Return 1 if sex_value is 'Male'
    if sex_value == 'Male':
        return 1
    
    # Return 0 if sex_value is 'Female'
    elif sex_value == 'Female':
        return 0
        
    # Return np.nan    
    else:
        return np.nan

# Apply the function to the sex column
'''Apply your recode_sex() function over tips.sex using the .apply() 
method to create a new column: 'sex_recode''''
tips['sex_recode'] = tips.sex.apply(recode_sex)


####Lambda Functions####
#Lambda functions allow your to clean your data more effectively with simple 1 line functions

# Write the lambda function using replace
'''Use the .replace() method inside a lambda function to remove the dollar sign from the 'total_dollar' column of tips.
You need to specify two arguments to the .replace() method: The string to be replaced ('$'), and the string to replace it by ('').
Apply the lambda function over tips.total_dollar.'''
tips['total_dollar_replace'] = tips.total_dollar.apply(lambda x: x.replace('$', ''))

# Write the lambda function using regular expressions
'''Use a regular expression to remove the dollar sign from the 'total_dollar' column of tips.
The pattern has been provided for you: It is the first argument of the re.findall() function.
Complete the rest of the lambda function and apply it over tips.total_dollar.'''
tips['total_dollar_re'] = tips.total_dollar.apply(lambda x: re.findall('\d+\.\d+', x))

# Print the head of tips
print(tips.head())


####Dropping Duplicate Data####

# Create the new DataFrame: tracks
tracks = billboard[['year', 'artist', 'track', 'time']]

# Print info of tracks
print(tracks.info())

# Drop the duplicates: tracks_no_duplicates
'''Drop duplicate rows from tracks using the .drop_duplicates() method. 
Save the result to tracks_no_duplicates'''
tracks_no_duplicates = tracks.drop_duplicates()

# Print info of tracks_no_duplicates
print(tracks_no_duplicates.info())


####Filling Missing Data####
#Mean Replace

# Calculate the mean of the Ozone column: oz_mean
oz_mean = airquality.Ozone.mean()

# Replace all the missing values in the Ozone column with the mean
airquality['Ozone'] = airquality.Ozone.fillna(oz_mean)

# Print the info of airquality
print(airquality.info())


####Testing Data to Check for Missing Data with Asserts####
'''.all() method together with the .notnull() DataFrame method to check for missing values 
in a column. The .all() method returns True if all values are True. When used on a DataFrame, 
it returns a Series of Booleans - one for each column in the DataFrame. So if you are using it 
on a DataFrame, like in this exercise, you need to chain another .all() method so that you return 
only one True or False value. When using these within an assert statement, nothing will be returned 
if the assert statement is true: This is how you can confirm that the data you are checking are valid.

Note: You can use pd.notnull(df) as an alternative to df.notnull().'''

# Assert that there are no missing values
'''Write an assert statement to confirm that there are no missing values in ebola.
Use the pd.notnull() function on ebola (or the .notnull() method of ebola) and chain two .all() 
methods (that is, .all().all()). The first .all() method will return a True or False for each 
column, while the second .all() method will return a single True or False.'''
assert pd.notnull(ebola).all().all()

# Assert that all values are >= 0
'''Write an assert statement to confirm that all values in ebola are greater than or equal to 0.
Chain two all() methods to the Boolean condition (ebola >= 0).'''
assert (ebola >= 0).all().all()
