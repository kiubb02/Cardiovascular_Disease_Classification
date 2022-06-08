# main.py
#######################################################
#                         IMPORTS                     #
#######################################################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from matplotlib.lines import Line2D

colors = ['#99ff99','#ffcc99']

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

print(f"Number of columns: {df.shape[1]}")
print(f"Number of rows: {df.shape[0]}")

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
df['age'] = round(df['age'] / 365.25)
df = df.astype({'age': int})

# convert weight table to int => no floats needed
df = df.astype({'weight': int})

# check for outliers

print("After Data Cleaning:   ")
print(df.head())
print(df.info())
print(df.describe())

# get correlations of the columns
print(f"Correlation of the columns and cardio\n{df.corr()['cardio'].sort_values(ascending=False)}")
# we see age and cholesterol have the biggest correlation with the cardio column

#######################################################
#                   VISUALIZE DATA                    #
#######################################################

# boxplot for age and cardiovascular disease
fig, ax = plt.subplots(ncols=2, figsize=(20, 10))
plt.tight_layout(pad=18)
sb.boxplot(data=df, x='cardio', y='age', ax=ax[0])
sb.boxplot(data=df, x='cardio', y='cholesterol', ax=ax[1])
ax[0].title.set_text('Age')
ax[0].set_xticklabels(['No-cardio', 'Cardio'])
ax[0].set_xlabel("")
ax[1].title.set_text('Cholesterol')
ax[1].set_xticklabels(['No-cardio', 'Cardio'])
ax[1].set_xlabel("")
plt.show()
plt.clf()


# pie charts
custom_lines = [Line2D([0], [0], color=colors[0], lw=4),
                Line2D([0], [0], color=colors[1], lw=4)]

data = df["gender"].value_counts()
ax = data.plot(kind="pie", autopct='%1.1f%%', shadow=True, explode=[0.05, 0.05], colors=colors, legend=True, title='Cardiovascular patients gender percentage', ylabel='', labeldistance=None)
ax.legend(custom_lines, ['Female', 'Male'], bbox_to_anchor=(1, 1.02), loc='upper left')
plt.show()
plt.clf()

data = df["cardio"].value_counts()
ax = data.plot(kind="pie", autopct='%1.1f%%', shadow=True, explode=[0.05, 0.05], colors=colors, legend=True, title='Cardiovascular patients percentage', ylabel='', labeldistance=None)
ax.legend(custom_lines, ['Not identified', 'Identified'], bbox_to_anchor=(1, 1.02), loc='upper left')
plt.tight_layout()
plt.show()
plt.clf()

# density plots
df['age'].plot.density(color=colors[0])
plt.title('Density plot for Age')
plt.show()
plt.clf()

df['cholesterol'].plot.density(color=colors[0])
plt.title('Density plot for Cholesterol')
plt.show()
plt.clf()

df['weight'].plot.density(color=colors[0])
plt.title('Density plot for Weight')
plt.show()
plt.clf()

#######################################################
#                 DATA PREPROCESSING                  #
#######################################################



#######################################################
#                   MODEL BUILDING                    #
#######################################################
