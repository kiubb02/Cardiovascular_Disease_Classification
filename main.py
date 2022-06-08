# main.py
#######################################################
#                         IMPORTS                     #
#######################################################

import pandas as pd
import matplotlib

#######################################################
#                      IMPORT DATA                    #
#######################################################

df = pd.read_csv(r"data/cardio_train.csv", sep=';')

#######################################################
#                     EXPLORE DATA                    #
#######################################################

print("Before Data Cleaning:   ")
print(df.head())
# Print a concise summary of a DataFrame.
print(df.info())
# Generate descriptive statistics.
print(df.describe())

print(f"Number of columns: { df.shape[1] }")
print(f"Number of rows: { df.shape[0] }")

# Create a histogram
df.hist()
print(type(df))

#######################################################
#                      CLEAN DATA                     #
#######################################################

# remove duplicate rows
print(f"Sum of duplicates {df.duplicated().sum()}")

# drop the id column
df = df.drop('id', 1)

# we have no null values
# when column has 0 and 1 values => binary => true or false
# ==> no missing data which needs to be handled

# we have only numeric values
# ==> no data transformation

# age is in days => convert into years => more understandable
df['age'] = round(df['age']/365.25)
df = df.astype({'age': int})

# convert weight table to int => no floats needed
df = df.astype({'weight': int})

# check for outliers

print("After Data Cleaning:   ")
print(df.head())
print(df.info())
print(df.describe())

#######################################################
#                   VISUALIZE DATA                    #
#######################################################



