#!/usr/bin/env python

# Import person.py file and necessary libraries
import person as p
import numpy as np
import matplotlib.pyplot as plt

# Lists already given
list_of_names = ['Roger', 'Mary', 'Luisa', 'Elvis']
list_of_ages  = [23, 24, 19, 86]
list_of_heights_cm = [175, 162, 178, 182]

# New list
name_len = []

for name in list_of_names:
  print("The name {:} is {:} letters long".format(name, len(name)))
  # Push length of name into new list
  name_len.append(len(name))

#Print Name Lengths
print(name_len)

# Create dictionary of names to person object
people = {name : p.person(name, list_of_ages[list_of_names.index(name)], list_of_heights_cm[list_of_names.index(name)]) for name in list_of_names}
print(people)

# Convert Lists to Numpy Arrays
np_ages = np.array(list_of_ages)
np_heights = np.array(list_of_heights_cm)

# Print Means
age_mean = np.mean(np_ages)
height_mean = np.mean(np_heights)
print("The mean age is "+ str(age_mean) +" and the height mean is " + str(height_mean))

# Plot ages and heights
plt.plot(np_ages, np_heights, 'ro')
plt.title("People Ages vs Heights")
plt.xlabel("Ages (yrs)")
plt.ylabel("Heights (cm)")
plt.grid(color='black')
plt.show()
