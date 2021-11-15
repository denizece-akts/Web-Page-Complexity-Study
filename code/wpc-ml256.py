import numpy as np
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import xgboost as xg
from sklearn import svm


imgs_tr = glob("./trainval/*.png")
imgs_test = glob("./test/*.png")

"""
"very basic" = range(5,20) : 1, 
"basic" = range(20,40) : 2, 
"moderate" = range(40,60) : 3, 
"complex" = range(60,80) : 4, 
"very complex" = range(80,100) : 5}
"""

# TRAIN & VALIDATION DATA

train_data = np.loadtxt('train_data256.csv', delimiter=',')

train = pd.DataFrame(columns=["ImageName", "Features", "Label"])
train['Features'] = train['Features'].astype(object)

trainval_labels = pd.read_csv("label-trainval.csv")

labels = []

for path in imgs_tr:
    
    img_name = path.split("\\")[1]
    mean_val = trainval_labels[trainval_labels["ImageName"] == img_name]["Mark"].values[0]
    
    if 5 < mean_val < 20:
        labels.append(1)
    elif 20 < mean_val < 40:
        labels.append(2)
    elif 40 < mean_val < 60:
        labels.append(3)
    elif 60 < mean_val < 80:
        labels.append(4)
    else:
        labels.append(5)

train["ImageName"] = [p.split("\\")[1] for p in imgs_tr]
train["Label"] = labels

for i in range(len(imgs_tr)):
    train.at[i, 'Features'] = train_data[i]

# TEST DATA

test_data = np.loadtxt('test_data256.csv', delimiter=',')

test = pd.DataFrame(columns=["ImageName", "Features", "Label"])
test['Features'] = test['Features'].astype(object)

test_labels = pd.read_csv("label-test.csv")

label_test = []

for path in imgs_test:
    
    img_name = path.split("\\")[1]
    mean_val = test_labels[test_labels["ImageName"] == img_name]["Mark"].values[0]
    
    if 5 < mean_val < 20:
        label_test.append(1)
    elif 20 < mean_val < 40:
        label_test.append(2)
    elif 40 < mean_val < 60:
        label_test.append(3)
    elif 60 < mean_val < 80:
        label_test.append(4)
    else:
        label_test.append(5)
        

test["ImageName"] = [p.split("\\")[1] for p in imgs_test]
test["Label"] = label_test

for i in range(len(imgs_test)):
    test.at[i, 'Features'] = test_data[i]


# Train data is splitted as %70 train, %30 val

X_train, X_val, y_train, y_val = train_test_split(train["Features"].values, 
                                                    train["Label"].values, 
                                                    test_size=0.30, 
                                                    random_state=42, 
                                                    shuffle=True)

X_train = np.array([np.array([np.array(i) for i in xtr]) for xtr in X_train])
y_train = np.array(y_train)

X_val = np.array([np.array([np.array(i) for i in xv]) for xv in X_val])
y_val = np.array(y_val)

X_test = np.array([np.array([np.array(i) for i in xt]) for xt in test["Features"].values])
y_test = np.array(test["Label"].values)


# RANDOM FOREST

rf = RandomForestClassifier(n_estimators = 1000, random_state = 42)
rf.fit(X_train, y_train)

predictions = rf.predict(X_val)
print("Random Forest algorithm result: ", rf.score(X_val, y_val))

print(classification_report(y_val, predictions))
print("Accuracy:", accuracy_score(y_val, predictions))

test_predictions = rf.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, test_predictions))


# XGBOOST

xgb = xg.XGBClassifier()
xgb.fit(X_train, y_train)

preds = xgb.predict(X_val)
print("XGBoost algorithm result: ", xgb.score(X_val, y_val))

print(classification_report(y_val, preds))
print("Accuracy:", accuracy_score(y_val, preds))

test_preds = xgb.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, test_preds))


# SVM

model = svm.SVC(kernel='rbf')   # Gaussian Kernel
model.fit(X_train, y_train)

y_pred = model.predict(X_val)
print("SVM algorithm result: ", model.score(X_val, y_val))

print(classification_report(y_val, y_pred))
print("Accuracy:", accuracy_score(y_val, y_pred))

test_y_pred = model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, test_y_pred))