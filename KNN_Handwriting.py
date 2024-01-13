import time
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import statistics
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from collections import Counter
import math,random

class KNN:

    def __init__(self,k,X_train,y_train):
        self.k = k
        self.X_train = X_train
        self.y_train = y_train


    def predictions(self,x_test):
        prediction=[]

        for x in x_test:
            prediction.append(self.predict_helper(x))

        return prediction

    def predict_helper(self,x):
        #calculate euclidean distance for every row in X_training set
        euc_distances=list()
        for x_train in self.X_train:
            #print("value of x-train is ",x_train)
            euc_distances.append(euclideandistance(x,x_train))


        k_indexes=np.argsort(euc_distances)[:self.k]
        #print("k_indexes are",k_indexes)
        k_targets=[self.y_train[i] for i in k_indexes]
        #print("k_targets are",k_targets)
        most_voted=Counter(k_targets).most_common(1)
        #print("most voted are",most_voted[0][0])
        return most_voted[0][0]



def load_dataset():
    digits = datasets.load_digits(return_X_y=True)
    digits_dataset_x = digits[0]
    digits_dataset_y = digits[1]
    return digits_dataset_x, digits_dataset_y

def k_fold_split(x, y, k):
    skfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    train_indices = []
    test_indices = []
    for train_index, test_index in skfold.split(x, y):
        train_indices.append(train_index)
        test_indices.append(test_index)
    return train_indices, test_indices

def normalize_features(dataset, train_dataset_min, train_dataset_max):
    normalized_dataset = (dataset - train_dataset_min) / (train_dataset_max - train_dataset_min)
    return normalized_dataset


def euclideandistance(r1, r2):
    dist=0.0
    no_attributes = len(r1)-1
    #print("no of attributes",no_attributes)
    for i in range(no_attributes):
        dist += (r1[i] - r2[i])**2
        #print(dist)
    d = math.sqrt(dist)
    #print(d)
    return d

def display_graph(Accuracy,std_dev,Label,K):
    #plotting the graph for both testing and training
    plt.errorbar(K,Accuracy,ecolor='r',yerr=std_dev,marker='o',label=Label)
    plt.xlabel('Value of K ')
    plt.ylabel('Accuracy over K')
    plt.ylim(0.8, 1.0)
    plt.legend()
    plt.show()


def main():
    start_time = time.time()

    x, y = load_dataset()
    k_folds = 10
    train_indices, test_indices = k_fold_split(x, y, k_folds)
    print(len(train_indices))
    print(len(test_indices))
    train_accuracy_per_k = []
    train_f1_score_per_k = []
    train_stdev_accuracy_per_k = []
    test_accuracy_per_k = []
    test_f1_score_per_k = []
    test_stdev_accuracy_per_k = []
    K=[]
    for k in list(range(1, 52, 2)):
        print("########## K: ", k,"##############")
        K.append(k)
        train_accuracy = []
        test_accuracy = []
        train_f1scores = []
        test_f1scores = []

        for fold in range(k_folds):
            print("Fold: ", fold)
            x_train, x_test = x[train_indices[fold]], x[test_indices[fold]]
            y_train, y_test = y[train_indices[fold]], y[test_indices[fold]]
            x_train_min = x_train.min()
            x_train_max = x_train.max()
            X_train = normalize_features(x_train, x_train_min, x_train_max)
            x_test = normalize_features(x_test, x_train_min, x_train_max)
            Knnobj = KNN(k,X_train,y_train)
            #call the training dataset first
            euclidean_labels_predicted= Knnobj.predictions(X_train)
            #print(euclidean_labels_predicted)
            acc_list=[x==y for x,y in zip(euclidean_labels_predicted,y_train)]

            #print("No of Correct Predictions for training is",acc)
            accuracy=np.mean(acc_list)
            print("accuracy for training is",accuracy)
            train_accuracy.append(accuracy)
            fs=f1_score(y_train, euclidean_labels_predicted, average='macro')
            print("F1 score for training is",fs)
            train_f1scores.append(fs)
            #call the testing dataset first
            print("Calling the test data")
            euclidean_labels_predicted= Knnobj.predictions(x_test)
            #print(euclidean_labels_predicted)
            acc=[x==y for x,y in zip(euclidean_labels_predicted,y_test)]
            #print("No of Correct Predictions for testing is",acc)

            accuracy_test=np.mean(acc)
            print("accuracy for testing is",accuracy_test)
            test_accuracy.append(accuracy_test)
            fs_test=f1_score(y_test, euclidean_labels_predicted, average='macro')
            print("F1 score for testing is",fs_test)
            test_f1scores.append(fs_test)
        train_accuracy_per_k.append(np.mean(train_accuracy))
        train_f1_score_per_k.append(np.mean(train_f1scores))
        train_stdev_accuracy_per_k.append(statistics.stdev(train_accuracy))
        test_accuracy_per_k.append(np.mean(test_accuracy))
        test_f1_score_per_k.append(np.mean(test_f1scores))
        test_stdev_accuracy_per_k.append(statistics.stdev(test_accuracy))

    end_time = time.time()
    print('Total time taken', end_time - start_time)
    print("train_accuracy_per_k: ", train_accuracy_per_k)
    print("train_stdev_accuracy_per_k: ", train_stdev_accuracy_per_k)
    print("train_f1_score_per_k: ", train_f1_score_per_k)
    print("test_accuracy_per_k: ", test_accuracy_per_k)
    print("test_stdev_accuracy_per_k: ", test_stdev_accuracy_per_k)
    print("test_f1_score_per_k: ", test_f1_score_per_k)
    display_graph(train_accuracy_per_k,train_stdev_accuracy_per_k,"Training Accuracy over K",K)
    display_graph(test_accuracy_per_k,test_stdev_accuracy_per_k,"Testing Accuracy over K",K)



if __name__ == "__main__":
    main()