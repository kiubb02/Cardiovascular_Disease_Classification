# main.py
#######################################################
#                         IMPORTS                     #
#######################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from matplotlib.lines import Line2D
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import tree
import xgboost as xgb

colors = ['#99ff99', '#ffcc99']
convert = 1

#######################################################
#                      IMPORT DATA                    #
#######################################################
# before we start lets import the data we are going to work with

df = pd.read_csv(r"data/cardio_train.csv", sep=';')

#######################################################
#                     EXPLORE DATA                    #
#######################################################

# now we want to explore the data
# we want to get to know it

print("Before Data Cleaning:   ")

# 1.
print(df.head())
# here we see that age is a bigger integer => age is in days which is not really redable
# 2.
# Print a concise summary of a DataFrame.
print(df.info())
# in that we see that our data has no null values
# 3.
# Generate descriptive statistics.
print(df.describe())

print(f"Number of columns: {df.shape[1]}")
print(f"Number of rows: {df.shape[0]}")

#######################################################
#                      CLEAN DATA                     #
#######################################################

# remove duplicate rows
print(f"Sum of duplicates {df.duplicated().sum()}")
# since we have 0 duplicates there is nothing to remove

# drop the id column => it is not needed
df = df.drop('id', 1)

# we have no null values
# when column has 0 and 1 values => binary => true or false
# ==> no missing data which needs to be handled

# we have only numeric values
# ==> no data transformation

# age is in days => convert into years => more understandable
df['age'] = round(df['age'] / 365.25)
df = df.astype({'age': int})

# lets check if weight has only whole numbers or not , then we can convert it
for index, row in df.iterrows():
    if not row['weight'].is_integer():
        convert = 0

# convert weight table to int => no floats needed
if convert == 1:
    df = df.astype({'weight': int})

# check for outliers

print("\nAfter Data Cleaning:   ")
print(df.head())
print(df.info())
print(df.describe())

# get correlations of the columns
print(f"Correlation of the columns and cardio\n{df.corr()['cardio'].sort_values(ascending=False)}")
# we see that age, cholesterol and weight are the top 3 columns in sense of correlating with the cardio column

#######################################################
#                   VISUALIZE DATA                    #
#######################################################

# boxplot for age and cardiovascular disease
# boxplot for weight and cardiovascular disease
fig, ax = plt.subplots(ncols=2, figsize=(20, 10))
plt.tight_layout(pad=18)
sb.boxplot(data=df, x='cardio', y='age', ax=ax[0])
sb.boxplot(data=df, x='cardio', y='weight', ax=ax[1])
ax[0].title.set_text('Age')
ax[0].set_xticklabels(['No-cardio', 'Cardio'])
ax[0].set_xlabel("")
ax[1].title.set_text('Weight')
ax[1].set_xticklabels(['No-cardio', 'Cardio'])
ax[1].set_xlabel("")
plt.show()
plt.clf()

# pie charts
# the pie charts are made for a general overview ==> why did we choose those?
custom_lines = [Line2D([0], [0], color=colors[0], lw=4),
                Line2D([0], [0], color=colors[1], lw=4)]

# chose to plot gender to check on the data
# Cardiovascular disease develops 7 to 10 years later in women than in men and is still the major cause of death in
# women over the age of 65 years.
# Men generally develop CVD at a younger age and have a higher propensity of developing coronary heart disease (CHD)
# than women. Women, in contrast, are at a higher risk of stroke, which often occurs at older age.
data = df["gender"].value_counts()
ax = data.plot(kind="pie", autopct='%1.1f%%', shadow=True, explode=[0.05, 0.05], colors=colors, legend=True,
               title='Cardiovascular patients gender percentage', ylabel='', labeldistance=None)
ax.legend(custom_lines, ['Female', 'Male'], bbox_to_anchor=(1, 1.02), loc='upper left')
plt.show()
plt.clf()

# chose to plot cardio to check on the data
data = df["cardio"].value_counts()
ax = data.plot(kind="pie", autopct='%1.1f%%', shadow=True, explode=[0.05, 0.05], colors=colors, legend=True,
               title='Cardiovascular patients percentage', ylabel='', labeldistance=None)
ax.legend(custom_lines, ['Not identified', 'Identified'], bbox_to_anchor=(1, 1.02), loc='upper left')
plt.tight_layout()
plt.show()
plt.clf()

data = df["smoke"].value_counts()
ax = data.plot(kind="pie", autopct='%1.1f%%', shadow=True, explode=[0.05, 0.05], colors=colors, legend=True,
               title='Smokers percentage', ylabel='', labeldistance=None)
ax.legend(custom_lines, ['None smokers', 'Smokers'], bbox_to_anchor=(1, 1.02), loc='upper left')
plt.tight_layout()
plt.show()
plt.clf()

data = df["alco"].value_counts()
ax = data.plot(kind="pie", autopct='%1.1f%%', shadow=True, explode=[0.05, 0.05], colors=colors, legend=True,
               title='Drinkers percentage', ylabel='', labeldistance=None)
ax.legend(custom_lines, ['None drinkers', 'Drinkers'], bbox_to_anchor=(1, 1.02), loc='upper left')
plt.tight_layout()
plt.show()
plt.clf()

data = df["active"].value_counts()
ax = data.plot(kind="pie", autopct='%1.1f%%', shadow=True, explode=[0.05, 0.05], colors=colors, legend=True,
               title='Activity percentage', ylabel='', labeldistance=None)
ax.legend(custom_lines, ['Active', 'Not active'], bbox_to_anchor=(1, 1.02), loc='upper left')
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

# Encoding categorical data => already done
# no data transformation needed

# feature selection / dimensionality reduction

# manual
# remove features that do not correlate with cardiovascular disease
# The most important behavioural risk factors of heart disease and stroke are unhealthy diet, physical inactivity,
# tobacco use and harmful use of alcohol. The effects of behavioural risk factors may show up in individuals as raised
# blood pressure, raised blood glucose, raised blood lipids, and overweight and obesity.

# overweight/obesity => drop tables weight and height and create a bmi column
df.insert(5, 'bmi', round((df['weight'] / (df['height'] / 100) ** 2), 2))
df = df.drop(columns=['height', 'weight'])


# Men generally develop CVD at a younger age and have a higher propensity of developing coronary heart disease (CHD)
# than women. ==> do not remove gender

# blood pressure => calculate out of ap_hi and ap_lo and then drop those tables
# normal ... -1
# elevated ... 0
# high 1 ... 1
# high 2 ... 2
# high 3 ... 3

def blood_pressure(x, y):
    if x <= 120 and y <= 80:
        return -1
    elif x <= 129 and y <= 80:
        return 0
    elif x <= 139 or y <= 89:
        return 1
    elif x <= 180 or y <= 120:
        return 2
    elif x > 180 or y > 120:
        return 3
    else:
        return None


df.insert(8, "bp", df.apply(lambda row: blood_pressure(row['ap_hi'], row['ap_lo']), axis=1))
df = df.drop(columns=['ap_hi', 'ap_lo'])

print("\nAfter Data Preprocessing: ")
print(df.head())
print(df.describe())
print(f"Correlation of the columns and cardio after data preprocessing\n{df.corr()['cardio'].sort_values(ascending=False)}")

# does correlation change ?
# yes we know can see that bp correlates much more to cardio than age (age was on the first place first)

#######################################################
#                   VISUALIZE DATA                    #
#######################################################

# we visualize the data further after we have processed it further
fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(20, 13))
plt.tight_layout(pad=3)

# first row all with tight relationship with cardio
df_bp = df.groupby('bp').mean()
sb.barplot(data=df_bp, x=df_bp.index, y='cardio', ax=ax[0][0])
df_cholesterol = df.groupby('cholesterol').mean()
sb.barplot(data=df_cholesterol, x=df_cholesterol.index, y='cardio', ax=ax[0][1])
ax[0][1].set_xticklabels(['normal', 'above normal', 'well above normal'])
df_gluc = df.groupby('gluc').mean()
sb.barplot(data=df_gluc, x=df_gluc.index, y='cardio', ax=ax[0][2])
ax[0][2].set_xticklabels(['normal', 'above normal', 'well above normal'])
ax[0][2].set_yticks(np.arange(0, 1.2, 0.1))
ax[0][2].set_yticklabels(np.arange(0, 120, 10))

# second row all with loose relationship with cardio
df_active = df.groupby('active').mean()
sb.barplot(data=df_active, x=df_active.index, y='cardio', ax=ax[1][0])
df_alco = df.groupby('alco').mean()
sb.barplot(data=df_alco, x=df_alco.index, y='cardio', ax=ax[1][1])
df_smoke = df.groupby('smoke').mean()
sb.barplot(data=df_smoke, x=df_smoke.index, y='cardio', ax=ax[1][2])

plt.setp(ax[:, :], ylabel='')
plt.setp(ax[:, 0], ylabel='Cardio Percentage')
plt.show()
plt.clf()

#######################################################
#                     CONCLUSION                      #
#######################################################

# Before we go to build a model lets sum up what we have learned from our data exploration
# 1. The correlation between smoking and the possibility of having a cardio vascular disease is low
# 2. The correlation between alcohol intake and the possibility of having a cardio vascular disease is low
# 3. The correlation between sports/activity and the possibility of having a cardio vascular disease is low
# 4. The relationship between the bmi and cardiovascular is really strong , thus the higher the bmi the more possible
# the probability of having a cardiovascular disease
# 5. Other major relationships to cardiovascular disease have the following features: cholesterol level, blood pressure
# and glucose level
# 6. All major relationships are proportional ( the higher x , the more likely y) => can be seen in the visualization


#######################################################
#                   MODEL BUILDING                    #
#######################################################
# in our case we need classification models:
# 1. KNN classification
# 2. Classification trees

#######################################################
#                 KNN CLASSIFICATION                  #
#######################################################
# We have to ...
# 1. Split the Data in test, train and validation = > Monte Carlo Cross-Validation or K-Fold Cross Validation
# 2. Hyper Parameter Tuning (hyper Parameter k)
# 3. Compare the prediction and get a performance evaluation of the model

# get features and target
X = df.iloc[:, :-1].to_numpy()
y = df['cardio'].to_numpy()

# ------------------------------------------------------------------------------- #
#                              HYPER PARAMETER TUNING                             #
# ------------------------------------------------------------------------------- #
# k_range = range(80, 100)
# k_scores = []

# for k in k_range:
#    knn = KNeighborsClassifier(n_neighbors=k)
#    scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
#    k_scores.append(scores.mean())

# plt.plot(k_range, k_scores)
# plt.xlabel('Value of K for KNN')
# plt.ylabel('Cross-Validation Accuracy')
# plt.show()

# now we can see which k would be the best
# we tried various spaces
# the best space was between 80-100 range for k in case of accuracy and speed
# print(np.argmax(k_scores))
# print(k_range[np.argmax(k_scores)])
# ------------------------------------------------------------------------------- #

# now lets actually train ad use the model and get all the data to it
k = 81

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print('accuracy: ', knn.score(X_test, y_test))

false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(y_test, y_pred)

print('roc_auc_score for KNN: ', roc_auc_score(y_test, y_pred))

plt.subplots(1, figsize=(10, 10))
plt.title('ROC - KNN')
plt.plot(false_positive_rate1, true_positive_rate1)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0], c=".7"), plt.plot([1, 1], c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
plt.clf()

#######################################################
#                        XGBOOST                      #
#######################################################

xgb_train_data = xgb.DMatrix(X_train, label=y_train)
xgb_test_data = xgb.DMatrix(X_test, label=y_test)

xgb_params = {
    "objective": "binary:hinge",
    'eval_metric': "mae"
}
evals = [(xgb_train_data, 'train'), (xgb_test_data, 'test')]

gbm = xgb.train(
    xgb_params,
    xgb_train_data,
    num_boost_round=100,
    early_stopping_rounds=10,
    evals=evals,
)

y_pred_xgb = gbm.predict(xgb_test_data)
acc = accuracy_score(y_test, y_pred_xgb)
print("Accuracy : ", acc)

_, ax = plt.subplots(figsize=(12, 4))
xgb.plot_importance(gbm,
                    ax=ax,
                    importance_type='gain',
                    show_values=False)
plt.show()
plt.clf()

# Explanation:
# f0 ... age
# f1 ... gender
# f2 ... bmi
# f3 ... cholesterol
# f4 ... glucose
# f5 ... smoke
# f6 ..- blood pressure ==> as said before and seen in graph : biggest impact
# f7 ... alcohol
# f8 ... active

false_positive_rate_xgb, true_positive_rate_xgb, threshold2 = roc_curve(y_test, y_pred_xgb)

print('roc_auc_score for XGB: ', roc_auc_score(y_test, y_pred_xgb))

plt.subplots(1, figsize=(10, 10))
plt.title('ROC - XGB')
plt.plot(false_positive_rate_xgb, true_positive_rate_xgb)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0], c=".7"), plt.plot([1, 1], c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
plt.clf()

# ------------------------------------------------------------------------------- #
#                                     CONCLUSION                                  #
#                                (after knn and xgb)                              #
# ------------------------------------------------------------------------------- #

# we cannot go over the accuracy of 72% => why is that ?
# a possibility might be the inclusion of outliers, which in our opinion are important to the case
# removing outliers still brough the same result (or a 73% result)

#######################################################
#                CLASSIFICATION TREES                 #
#######################################################

# ------------------------------------------------------------------------------- #
#                              HYPER PARAMETER TUNING                             #
# ------------------------------------------------------------------------------- #
#
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# train_accuracies = []
# test_accuracies = []
# depth_range = range(1, 15)
# for max_depth in depth_range:
#     decision_tree = DecisionTreeClassifier(max_depth=max_depth, random_state=0)
#     decision_tree.fit(X_train, y_train)
#     train_accuracies.append(decision_tree.score(X_train, y_train))
#     test_accuracies.append(decision_tree.score(X_test, y_test))
#
# plt.title('Training VS Test accuracy - max_depth')
# plt.plot(depth_range, train_accuracies, label='training accuracy')
# plt.plot(depth_range, test_accuracies, label='test accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('max_depth')
# plt.legend()
# plt.show()
# plt.clf()
#
#
# train_accuracies.clear()
# test_accuracies.clear()
# max_leaf_node_range = range(2, 80)
# for max_leaf_node in max_leaf_node_range:
#     decision_tree = DecisionTreeClassifier(max_depth=7, max_leaf_nodes=max_leaf_node, random_state=0)
#     decision_tree.fit(X_train, y_train)
#     train_accuracies.append(decision_tree.score(X_train, y_train))
#     test_accuracies.append(decision_tree.score(X_test, y_test))
#
# plt.title('Training VS Test accuracy - max_leaf_nodes')
# plt.plot(max_leaf_node_range, train_accuracies, label='training accuracy')
# plt.plot(max_leaf_node_range, test_accuracies, label='test accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('max_leaf_nodes')
# plt.legend()
# plt.show()
# plt.clf()
#
# dt = DecisionTreeClassifier(random_state=0)
# params = {
#     'max_depth': range(3, 8),
#     'max_leaf_nodes': range(31, 57),
#     'criterion': ['gini', 'entropy']
# }
# grid_search = GridSearchCV(estimator=dt,
#                            param_grid=params,
#                            cv=5, n_jobs=-1, verbose=1, scoring='accuracy')
#
# grid_search.fit(X_train, y_train)
# score_df = pd.DataFrame(grid_search.cv_results_)
# print(score_df.head())
# print(score_df.nlargest(3, 'mean_test_score'))
# print(score_df.nlargest(1, 'mean_test_score'))
# print(grid_search.best_estimator_)

decision_tree = DecisionTreeClassifier(max_depth=7, max_leaf_nodes=39, random_state=0)

decision_tree = decision_tree.fit(X_train, y_train)
y_pred_tree = decision_tree.predict(X_test)

text_representation = tree.export_text(decision_tree)
print(text_representation)

fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(decision_tree,
                   filled=True)
plt.show()
plt.clf()

decision_tree_train = decision_tree.fit(X_train, y_train)
y_pred_tree_train = decision_tree.predict(X_train)

acc = accuracy_score(y_test, y_pred_tree)
acc_train = accuracy_score(y_train, y_pred_tree_train)
print("Accuracy on training data: ", acc_train)
print("Accuracy on testing data: ", acc)

false_positive_rate_tree, true_positive_rate_tree, threshold3 = roc_curve(y_test, y_pred_tree)
print('roc_auc_score for Decision Tree: ', roc_auc_score(y_test, y_pred_tree))

plt.subplots(1, figsize=(10, 10))
plt.title('ROC - Decision Tree')
plt.plot(false_positive_rate_tree, true_positive_rate_tree)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0], c=".7"), plt.plot([1, 1], c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
plt.clf()

