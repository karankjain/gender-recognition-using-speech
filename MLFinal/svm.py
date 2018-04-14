import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from subprocess import check_output
from sklearn import linear_model
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_learning_curves

all_data = pd.read_csv("../input/voice.csv")

label_encoder = preprocessing.LabelEncoder()
all_data["label"] = label_encoder.fit_transform(all_data["label"])

rand_indices = np.random.permutation(len(all_data))
features = [feat for feat in all_data.columns if feat != "label"]

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

def svcc(c, of, svm_kernel):
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
    
    print(num_datapoints, len(train_data), len(test_data))
    print(features)
    #print (test_labels)

    svc = svm.SVC(kernel=svm_kernel, C=c,gamma= 'auto')
    svc.fit(train_data, train_labels)

    predictions = svc.predict(valid_data)
    print ('----------Validation Data-----------')
    print ('C: ' + str(c) + '\t Kernel: ' + svm_kernel) 
    print ('Accuracy Score:')
    print (accuracy_score(valid_labels, predictions))
    print ('Precision Score:')
    print (precision_score(valid_labels, predictions))
    print ('Recall Score:')
    print (recall_score(valid_labels, predictions))
    print ('Confusion Matrix:')
    print (confusion_matrix(valid_labels, predictions))

    plot_learning_curves(train_data, train_labels, valid_data, valid_labels, svc)
    plot_1 = plt

    predictions = svc.predict(test_data)
    print ('----------Test Data-----------')
    print ('C: ' + str(c) + '\t Kernel: ' + svm_kernel)
    print ('Accuracy Score:')
    print (accuracy_score(test_labels, predictions))
    print ('Precision Score:')
    print (precision_score(test_labels, predictions))
    print ('Recall Score:')
    print (recall_score(test_labels, predictions))
    print ('Confusion Matrix:')
    print (confusion_matrix(test_labels, predictions))

    plot_learning_curves(train_data, train_labels, test_data, test_labels, svc)
    plot_1.show()
    plt.show()

C_list = [1, 5, 10]
for i in C_list:
    svcc(i, False, 'linear')
    svcc(i, True, 'linear')
    svcc(i, False, 'rbf')
    svcc(i, True, 'rbf')
    
