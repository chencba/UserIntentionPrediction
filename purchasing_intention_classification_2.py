

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(10,10))
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE # pip install imbalanced-learn
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
import seaborn as sns

pd.set_option('display.max_columns', None)

path = r"D:/UCD/classes/Data Management & Mining/Classification/"

filename = "online_shoppers_intention.csv"

df = pd.read_csv(path + filename)

# transform of data types
df["OperatingSystems"] = df["OperatingSystems"].astype("category")
df["Browser"] = df["Browser"].astype("category")
df["Region"] = df["Region"].astype("category")
df["TrafficType"] = df["TrafficType"].astype("category")
df["Weekend"] = df["Weekend"].astype("category")
df["Revenue"] = df["Revenue"].astype("category")

"""
# feature selection - pearson's r
im = ax.imshow(df.corr(),cmap='plasma_r')
ax.set_xticks(np.arange(10)) 
ax.set_yticks(np.arange(10)) 
ax.set_xticklabels(["Administrative","Administrative_Duration","Informational",\
               "Informational_Duration","ProductRelated","ProductRelated_Duration",\
                   "BounceRates","ExitRates","PageValues","SpecialDay"], rotation="vertical")
ax.set_yticklabels(["Administrative","Administrative_Duration","Informational",\
               "Informational_Duration","ProductRelated","ProductRelated_Duration",\
                   "BounceRates","ExitRates","PageValues","SpecialDay"])
fig.colorbar(im,pad=0.03)
#sns.pairplot(df, hue="Revenue")
"""

df.drop(["SpecialDay", "TrafficType", "Weekend"], axis=1, inplace=True)

df["ProductRelated^Duration"] = df["ProductRelated"] * df["ProductRelated_Duration"]
df["Bounce^Exit"] = df["BounceRates"] * df["ExitRates"]

df.drop(["ProductRelated", "ProductRelated_Duration", "BounceRates", "ExitRates"], axis=1, inplace=True)

# dummy variable
col1 = df[["Month","OperatingSystems","Browser","Region",\
                "VisitorType","Revenue"]]

categorical = pd.get_dummies(col1, drop_first=True)


df = df[["Administrative","Administrative_Duration","Informational",\
               "Informational_Duration","ProductRelated^Duration",\
                   "Bounce^Exit","PageValues"]].join(categorical)

# get X and y
X = df.drop("Revenue_True", axis=1)
y = df["Revenue_True"]

"""
# feature selection - mutual information
selector = SelectKBest(mutual_info_classif, k=20)
selector.fit(X, y)
cols = selector.get_support(indices=True)
features_new = df.columns[cols]
print(features_new)

"""

# test set
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# cross validation
for i in np.arange(10):
    # train set & validation set
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2)
    
    # oversampling on train set
    smt = SMOTE()
    X_train, y_train = smt.fit_resample(X_train, y_train)
    
    # normalization on train set
    col2 = ["Administrative","Administrative_Duration","Informational",\
                   "Informational_Duration","ProductRelated^Duration",\
                       "Bounce^Exit","PageValues"]
    transformer = RobustScaler().fit(X_train[col2])
    X_numerical_train = pd.DataFrame(transformer.transform(X_train[col2]), \
                               columns=["Administrative","Administrative_Duration","Informational","Informational_Duration","ProductRelated^Duration","Bounce^Exit","PageValues"])
    
    # merge all features on train set
    X_train.drop(col2, axis=1, inplace=True)
    X_train = X_numerical_train.join(X_train)
    
    # normalization on validation set
    X_numerical_val = pd.DataFrame(transformer.transform(X_val[col2]), \
                               columns=["Administrative","Administrative_Duration","Informational","Informational_Duration","ProductRelated^Duration","Bounce^Exit","PageValues"])
    
    # merge all features on validation set
    X_val.drop(col2, axis=1, inplace=True)
    X_val.reset_index(inplace=True)
    X_val = X_numerical_val.join(X_val)
    X_val.drop("index", axis=1, inplace=True)
    
    """

    # decision tree
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    # naive bayes
    clf = GaussianNB()
    clf.fit(X_train, y_train)

    # SVM
    clf = SVC(gamma='auto')
    clf.fit(X_train, y_train)
  
    # KNN
    clf = KNeighborsClassifier(n_neighbors=4)
    clf.fit(X_train, y_train)

    # decision tree
    clf = DecisionTreeClassifier(criterion="entropy")
    clf.fit(X_train, y_train)
    #from sklearn import tree
    #tree.plot_tree(clf)
    
    # random forest
    clf = RandomForestClassifier(max_depth=4)
    clf.fit(X_train, y_train)
    """
    # logistic regreesion
    clf = LogisticRegression(max_iter=400)
    clf.fit(X_train, y_train)

    
    # Metrics
    prediction = clf.predict(X_val)
    
    print("result {}\n".format(i))
    results = classification_report(y_val, prediction, target_names=["With intention", "Without intention"])
    print(results)

# normalization on test set
X_numerical_test = pd.DataFrame(transformer.transform(X_test[col2]), \
                           columns=["Administrative","Administrative_Duration","Informational","Informational_Duration","ProductRelated^Duration","Bounce^Exit","PageValues"])

# merge all features on test set
X_test.drop(col2, axis=1, inplace=True)
X_test.reset_index(inplace=True)
X_test = X_numerical_test.join(X_test)
X_test.drop("index", axis=1, inplace=True)

print("result on test set")
result = classification_report(y_test, clf.predict(X_test), target_names=["With intention", "Without intention"])
print(result)