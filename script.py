import codecademylib3
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

#https://archive.ics.uci.edu/ml/machine-learning-databases/flags/flag.data
cols = ['name','landmass','zone', 'area', 'population', 'language','religion','bars','stripes','colours',
'red','green','blue','gold','white','black','orange','mainhue','circles',
'crosses','saltires','quarters','sunstars','crescent','triangle','icon','animate','text','topleft','botright']
df= pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/flags/flag.data", names = cols)

#variable names to use as predictors
var = [ 'red', 'green', 'blue','gold', 'white', 'black', 'orange', 'mainhue','bars','stripes', 'circles','crosses', 'saltires','quarters','sunstars','triangle','animate']

#Print number of countries by landmass, or continent
print(df.landmass.value_counts())

#Create a new dataframe with only flags from Europe and Oceania
df_36 = df[df['landmass'].isin([3,6])]

#Print the average vales of the predictors for Europe and Oceania
print(df_36.groupby('landmass')[var].mean().T)

#Create labels for only Europe and Oceania
labels = df_36["landmass"]

#Print the variable types for the predictors
print(df_36[var].dtypes)

#Create dummy variables for categorical predictors
data = pd.get_dummies(df_36[var])

#Split data into a train and test set
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.4, random_state=1)

#Fit a decision tree for max_depth values 1-20; save the accuracy score in acc_depth
depths = range(1, 21)
acc_depth = []

for i in depths:
  dtree = DecisionTreeClassifier(max_depth=i)
  dtree.fit(train_data, train_labels)
  acc_depth.append(dtree.score(test_data, test_labels))

#Plot the accuracy vs depth
plt.plot(depths, acc_depth)
plt.xlabel("Depth of a tree")
plt.ylabel("Accuracy")
plt.show()
plt.close()

#Find the largest accuracy and the depth this occurs
max_acc = np.max(acc_depth)
best_depth = depths[np.argmax(acc_depth)]

print(f"Best accuracy is {max_acc} for depth {best_depth}")

#Refit decision tree model with the highest accuracy and plot the decision tree
best_dtree = DecisionTreeClassifier(max_depth=best_depth)
best_dtree.fit(train_data, train_labels)

tree.plot_tree(best_dtree, feature_names=train_data.columns, class_names = ['Europe', 'Oceania'],
                filled=True)
plt.show()
plt.close()

#Create a new list for the accuracy values of a pruned decision tree.  Loop through
#the values of ccp and append the scores to the list
acc_pruned = []
cpp = np.logspace(-3, 0, num=20)
for i in cpp:
  dt_prune = DecisionTreeClassifier(random_state = 1, max_depth = best_depth, ccp_alpha=i)
  dt_prune.fit(train_data, train_labels)
  acc_pruned.append(dt_prune.score(test_data, test_labels))

#Plot the accuracy vs ccp_alpha
plt.plot(cpp, acc_pruned)
plt.show()
plt.close()

#Find the largest accuracy and the ccp value this occurs
max_acc_pruned = np.max(acc_pruned)
best_cpp = cpp[np.argmax(acc_pruned)]

print(f"Best accuracy is {max_acc_pruned} for cpp {best_cpp}")

#Fit a decision tree model with the values for max_depth and ccp_alpha found above
dt_last = DecisionTreeClassifier(random_state = 1, max_depth = best_depth, ccp_alpha=best_cpp)
dt_last.fit(train_data, train_labels)

#Plot the final decision tree
tree.plot_tree(dt_last, feature_names=train_data.columns, class_names = ['Europe', 'Oceania'],
                filled=True)
plt.show()
