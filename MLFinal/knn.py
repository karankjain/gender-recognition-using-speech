import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 
from mlxtend.plotting import plot_learning_curves
import operator


df = pd.read_csv('../input/voice.csv')

y=df.iloc[:,-1]
X=df.iloc[:, :-1]
X.head()

gender_encoder = LabelEncoder()
#Male=1, Female=0
y = gender_encoder.fit_transform(y)

#Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
train_data, test_data, train_labels, test_labels = train_test_split(X, y, test_size=0.3, random_state=1)

k_range=list(range(1,100))
acc_score=[]
for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k) 
    knn.fit(train_data,train_labels)
    y_pred=knn.predict(test_data)
    acc_score.append(accuracy_score(test_labels,y_pred))

k_values=list(range(1,100))
plt.plot(k_values,acc_score)
plt.xlabel('Value of k for knn')
plt.ylabel('Accuracy')
plt.show()

index, value = max(enumerate(acc_score), key=operator.itemgetter(1))
print("With k:" + str(index+1) + " we get most high accuracy of " + str(value))

def get_test_indices(total):
    one_third = int(total * 0.3)
    temp_indices_1 = list(range(int(one_third * 0.3), one_third-1))
    temp_indices_2 = list(range(one_third + (int(one_third * 0.3)), one_third*2 - 1 ) )
    temp_indices_3 = list(range((one_third*2) + (int(one_third * 0.3)), total ))
    return temp_indices_1 + temp_indices_2 + temp_indices_3

def get_train_indices(total):
    one_third = int(total * 0.3)
    temp_indices_1 = list(range(int(one_third * 0.3)+1))
    temp_indices_2 = list(range(one_third, one_third + (int(one_third * 0.3))))
    temp_indices_3 = list(range(one_third*2, (one_third*2) + (int(one_third * 0.3)) ))
    return temp_indices_1 + temp_indices_2 + temp_indices_3

def kNN(of):
    all_data = pd.read_csv("../input/voice.csv")

    label_encoder = LabelEncoder()
    all_data["label"] = label_encoder.fit_transform(all_data["label"])

    rand_indices = np.random.permutation(len(all_data))
    features = [feat for feat in all_data.columns if feat != "label"]
   
    if of:
        try:
            features.remove('modindx')
            features.remove('dfrange')
            features.remove('maxdom')
            features.remove('mindom')
            features.remove('meandom')
            features.remove('maxfun')
            features.remove('minfun')
            features.remove('mode')
            features.remove('kurt')
            features.remove('skew')
            features.remove('Q75')
        except:
            print()

    print(features)
    output = "label"
    num_datapoints = len(all_data)
    test_total = int(num_datapoints * 0.3)

    test_set_indices = get_test_indices(num_datapoints)
    test_indices = []
    valid_indices = []
    for i in test_set_indices:
        if(len(test_indices) < len(test_set_indices) * 0.5):
            test_indices.insert(len(test_indices), i)
        else:
            valid_indices.insert(len(valid_indices), i)
    train_indices = get_train_indices(num_datapoints)

    test_data = all_data[features].iloc[test_indices]
    valid_data = all_data[features].iloc[valid_indices]
    train_data = all_data[features].iloc[train_indices]

    test_labels = all_data[output].iloc[test_indices]
    valid_labels = all_data[output].iloc[valid_indices]
    train_labels = all_data[output].iloc[train_indices]

    knn = KNeighborsClassifier(n_neighbors=3)

    knn.fit(train_data, train_labels)
    y_pred = knn.predict(valid_data)
    print ('------Validation Data-------')
    print ('Accuracy Score:')
    print (accuracy_score(valid_labels, y_pred))
    print ('Precision Score:')
    print (precision_score(valid_labels, y_pred))
    print ('Recall Score:')
    print (recall_score(valid_labels, y_pred))
    print ('Confusion Matrix:')
    print (confusion_matrix(valid_labels, y_pred))

    plot_learning_curves(valid_data, valid_labels, test_data, test_labels, knn)
    plot_1 = plt

    
    knn.fit(train_data, train_labels)
    y_pred = knn.predict(test_data)
    print ('------Test Data-------')
    print ('Accuracy Score:')
    print (accuracy_score(test_labels, y_pred))
    print ('Precision Score:')
    print (precision_score(test_labels, y_pred))
    print ('Recall Score:')
    print (recall_score(test_labels, y_pred))
    print ('Confusion Matrix:')
    print (confusion_matrix(test_labels, y_pred))

    plot_learning_curves(train_data, train_labels, test_data, test_labels, knn)
    plot_1.show()
    plt.show()

kNN(False)
kNN(True)
