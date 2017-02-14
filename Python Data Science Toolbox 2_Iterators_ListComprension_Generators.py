#########################Iterators & Iterables##############################

#####Create an iterator example 1 #####
# Create a list of strings: flash
flash = ['jay garrick', 'barry allen', 'wally west', 'bart allen']

# Print each list item in flash using a for loop
for person in flash:
    print(person)

# Create an iterator for flash: superspeed
superspeed = iter(flash)

# Print each item from the iterator
print(next(superspeed))
print(next(superspeed))
print(next(superspeed))
print(next(superspeed))


#####Create an iterator example 2#####
# Create an iterator for range(3): small_value
small_value = iter(range(3))

# Print the values in small_value
print(next(small_value))
print(next(small_value))
print(next(small_value))

# Loop over range(3) and print the values
for num in range(3):
    print(num)

# Create an iterator for range(10 ** 100): googol
googol = iter(range(10 ** 100))

# Print the first 5 values from googol
print(next(googol))
print(next(googol))
print(next(googol))
print(next(googol))
print(next(googol))


#####Enumerate example#####
# Create a list of strings: mutants
mutants = ['charles xavier', 
            'bobby drake', 
            'kurt wagner', 
            'max eisenhardt', 
            'kitty pride']

# Create a list of tuples: mutant_list
mutant_list = list(enumerate(mutants))

# Print the list of tuples
print(mutant_list)

# Unpack and print the tuple pairs
for index1, value1 in enumerate(mutants):
    print(index1, value1)

# Change the start index
for index2, value2 in enumerate(mutants, start=1):
    print(index2, value2)


#########################List Comprehension##############################
# [[output expression] for iterator variable in iterable]

#####Simple List Comprehension#####
# Create list comprehension: squares
squares = [i**2 for i in range(0,10)]


#####Nested List Comprehension - create a matrix#####
# Create a 5 x 5 matrix using a list of lists: matrix
matrix = [[col for col in range(5)] for row in range(5)]

# Print the matrix
for row in matrix:
    print(row)


##########################Conditionals in List Comprehensions##########################
#[ output expression for iterator variable in iterable if predicate expression ]

####example 1: if#####
# Create a list of strings: fellowship
fellowship = ['frodo', 'samwise', 'merry', 'aragorn', 'legolas', 'boromir', 'gimli']

# Create list comprehension: new_fellowship
# Use 'member' as the iterator variable
# Only want strings 7 characters or more
new_fellowship = [member for member in fellowship if len(member) > 6]

# Print the new list
print(new_fellowship)


#####example 2: if-else#####
# Create a list of strings: fellowship
fellowship = ['frodo', 'samwise', 'merry', 'aragorn', 'legolas', 'boromir', 'gimli']

# Create list comprehension: new_fellowship
# Use 'member' as the iterator variable
# if the number of characters is >= 7, else replace it with an empty string
new_fellowship = [member if len(member) >= 7 else '' for member in fellowship]

# Print the new list
print(new_fellowship)


################################Dict Comprehension##################################
# curly braces {} instead of []. Members of the dictionary are created using a colon :, as in key:value.

#####Example 1#####
# Create a list of strings: fellowship
fellowship = ['frodo', 'samwise', 'merry', 'aragorn', 'legolas', 'boromir', 'gimli']

# Create dict comprehension: new_fellowship
# Use 'member' as the iterator variable
# The key is a string in fellowship
# The value is the length of the string
# Remember to use the Key : value in the output expression portion of the comprehension
new_fellowship = {member:len(member) for member in fellowship}

# Print the new list
print(new_fellowship)


##############################Generators############################################
# generators use () while comprehensions use {} or []
# List comprehension
fellow1 = [member for member in fellowship if len(member) >= 7]

# Generator expression
fellow2 = (member for member in fellowship if len(member) >= 7)


#####example 1#####
# Create generator object with values from 0 to 30: result
result = (num for num in range(31))

# Print the first 5 values
print(next(result))
print(next(result))
print(next(result))
print(next(result))
print(next(result))

# Print the rest of the values
for value in result:
    print(value)


#####example 2#####
# Create a list of strings: lannister
lannister = ['cersei', 'jaime', 'tywin', 'tyrion', 'joffrey']

# Create a generator object: lengths
# A generator expression that will generate the lengths of each string in lannister
# Use person as the iterator variable
lengths = (len(person) for person in lannister)

# Iterate over and print the values in lengths
for value in lengths:
    print(value)


#####Build a generator function#####
# Uses the the keyword yield instead of return
# Yield a series of values, instead of returning a single value

# Create a list of strings
lannister = ['cersei', 'jaime', 'tywin', 'tyrion', 'joffrey']

# Define generator function get_lengths
def get_lengths(input_list):
    """Generator function that yields the
    length of the strings in input_list."""

    # Yield the length of a string
    for person in input_list:
        yield len(person)

# Print the values generated by get_lengths()
for value in get_lengths(lannister):
    print(value)