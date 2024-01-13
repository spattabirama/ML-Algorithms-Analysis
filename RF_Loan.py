import math
import random
import numpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import sklearn.metrics as metrics


class Node:
    def __init__(self):
        self.name = None
        self.is_leaf = False
        self.label = None
        self.branches = []
        self.condition = None
        self.mean_val = 0.0


class PerformanceMetrics:
    def __init__(self):
        self.accuracy = 0.0
        self.precision = 0.0
        self.recall = 0.0
        self.beta = 1.0
        self.f_score = 0.0


def read_csv(filename):
    dataset = pd.read_csv(filename, sep='[\t,]', engine='python')
    return dataset


def k_fold_split(df, k, class_column_idx, class_column_name):
    class_val = np.unique(df.iloc[:, class_column_idx])
    df_per_class = {}
    split_per_class = {}
    k_fold_df = []
    for val in class_val:
        df_per_class[val] = df[df[class_column_name] == val]
        split_per_class[val] = round(len(df_per_class[val].index) / k)
    for i in range(k):
        i_df = pd.DataFrame()
        for cls in df_per_class:
            if i == k - 1:
                smp = df_per_class[cls]
            else:
                smp = df_per_class[cls].sample(split_per_class[cls])
            data = [i_df, smp]
            i_df = pd.concat(data)
            df_per_class[cls] = df_per_class[cls].drop(smp.index)
        k_fold_df.append(i_df)
    return k_fold_df


def create_bootstraps(train, n):
    b_list = []
    for i in range(n):
        train_idx = train.index
        b_idx = np.random.choice(train_idx, replace=True, size=len(train.index))
        b = train.iloc[b_idx, :]
        b_list.append(b)
    return b_list


def calculate_entropy(col):
    value, count = np.unique(col, return_counts=True)
    prob = count/count.sum()
    entropy = 0
    for p in prob:
        entropy = entropy + (-p * np.log10(p))
    return entropy


def label_subsets(att, y_train):
    att_values = np.unique(att)
    subsets = {}
    for val in att_values:
        subsets[val] = []
        for i in range(len(att)):
            if att.iat[i] == val:
                subsets[val].append(y_train.iat[i])
    return subsets


def numerical_label_subsets(att, y_train):
    mean_att_val = np.mean(att)
    subsets = {}
    subsets[mean_att_val] = {}
    subsets[mean_att_val]["<="] = []
    subsets[mean_att_val][">"] = []
    for i in range(len(att)):
        if att.iat[i] <= mean_att_val:
            subsets[mean_att_val]["<="].append(y_train.iat[i])
        else:
            subsets[mean_att_val][">"].append(y_train.iat[i])
    return subsets


def categorical_info_gain(att, y_train, target_entropy):
    subsets = label_subsets(att, y_train)
    att_entropy = 0.0
    for key in subsets:
        ent = calculate_entropy(subsets[key])
        att_entropy += (len(subsets[key])/len(att))*ent
    return target_entropy - att_entropy


def numerical_info_gain(att, y_train, target_entropy):
    subsets = numerical_label_subsets(att, y_train)
    att_info_gain = {}
    for mean_val, subset in subsets.items():
        mean_entropy = 0.0
        for condition, y_subset in subset.items():
            ent = calculate_entropy(y_subset)
            mean_entropy += (len(y_subset) / len(att)) * ent
        att_info_gain[mean_val] = target_entropy - mean_entropy
    return max(att_info_gain, key=att_info_gain.get), max(att_info_gain.values())


def find_max_info_att(df, class_column_index, target_entropy, df_type):
    attribute_info_gain = {}
    max_info_gain_condition = {}
    feature_columns_index = list(range(0, len(df.columns)))
    feature_columns_index.pop(class_column_index)
    rand_att_count = round(math.sqrt(len(df.columns)-1))
    rand_col_indexes = random.sample(feature_columns_index, rand_att_count)
    for j in rand_col_indexes:
        if df_type == 'categorical':
            info_gain = categorical_info_gain(df.iloc[:, j], df.iloc[:, class_column_index], target_entropy)
        else:
            mean_val, info_gain = numerical_info_gain(df.iloc[:, j], df.iloc[:, class_column_index], target_entropy)
            max_info_gain_condition[mean_val] = info_gain
        attribute_info_gain[df.columns[j]] = info_gain
    if df_type == 'categorical':
        return max(attribute_info_gain, key=attribute_info_gain.get), max(attribute_info_gain.values())
    else:
        return max(attribute_info_gain, key=attribute_info_gain.get), max(max_info_gain_condition, key=max_info_gain_condition.get), max(attribute_info_gain.values())


def dev_tree(df, class_column, node, min_size_for_split, df_type):
    target_entropy = calculate_entropy(df.iloc[:, class_column])
    node.label = df.iloc[:, class_column].mode().to_numpy()
    if target_entropy == 0.0:
        node.is_leaf = True
    elif len(df.index) < min_size_for_split:
        node.is_leaf = True
    if not node.is_leaf:
        if df_type == 'categorical':
            max_info_attribute, att_info_gain = find_max_info_att(df, class_column, target_entropy, df_type)
            node.name = max_info_attribute
            for att_val in np.unique(df[node.name]):
                att_val_dataset = df[df[node.name] == att_val]
                child = Node()
                child.condition = att_val
                node.branches.append(dev_tree(att_val_dataset, class_column, child, min_size_for_split, df_type))
        else:
            max_info_attribute, max_info_mean, att_info_gain = find_max_info_att(df, class_column, target_entropy, df_type)
            node.name = max_info_attribute
            att_val_dataset1 = df[df[node.name] <= max_info_mean]
            att_val_dataset2 = df[df[node.name] > max_info_mean]
            if len(att_val_dataset1.index) == 0 or len(att_val_dataset2.index) == 0:
                node.is_leaf = True
            if not node.is_leaf:
                for i in range(2):
                    if i == 0:
                        child = Node()
                        child.condition = '<='
                        child.mean_val = max_info_mean
                        node.branches.append(
                            dev_tree(att_val_dataset1, class_column, child, min_size_for_split, df_type))
                    else:
                        child = Node()
                        child.condition = '>'
                        child.mean_val = max_info_mean
                        node.branches.append(
                            dev_tree(att_val_dataset2, class_column, child, min_size_for_split, df_type))
    return node


def predict(tree, instance, df_type):
    if tree.is_leaf:
        if len(tree.label) > 0:
            prediction = tree.label[0]
        else:
            prediction = None
        return prediction
    else:
        node_name = tree.name
        feature_value = instance[node_name]

        if df_type == 'categorical':
            for branch in tree.branches:
                if branch.condition == feature_value:
                    prediction = predict(branch, instance, df_type)
                    if prediction is None:
                        prediction = branch.label[0]
                    return prediction
        else:
            for branch in tree.branches:
                if feature_value <= branch.mean_val and branch.condition == '<=':
                    prediction = predict(branch, instance, df_type)
                    return prediction
                elif feature_value > branch.mean_val and branch.condition == '>':
                    prediction = predict(branch, instance, df_type)
                    return prediction


def evaluate_random_forest(n_trees, test, class_column_name, df_type):
    n_trees_test_result = []
    for tree in n_trees:
        test_result = []
        for i in range(len(test)):
            result = predict(tree, test.iloc[i], df_type)
            test_result.append(result)
        n_trees_test_result.append(np.array(test_result))
    n_trees_test_result = np.stack(n_trees_test_result, axis=1)

    actual = test[class_column_name].to_numpy()
    actual = actual.astype(np.int64)
    max_prediction = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=n_trees_test_result)

    actual = np.array(actual)
    max_prediction = np.array(max_prediction)
    pm = PerformanceMetrics()
    pm.accuracy = metrics.accuracy_score(actual, max_prediction)
    pm.precision = metrics.precision_score(actual, max_prediction)
    pm.recall = metrics.recall_score(actual, max_prediction)
    pm.f_score = metrics.f1_score(actual, max_prediction)
    print(" Accuracy: ", pm.accuracy, "\n", "Precision: ", pm.precision, "\n", "Recall: ", pm.recall, "\n", "F_Score:", pm.f_score)
    return pm


def plot_graph(full_pm_matrix, n_vals, k):
    accuracy_per_n = []
    precision_per_n = []
    recall_per_n = []
    f_score_per_n = []
    for j in range(full_pm_matrix.shape[1]):
        accuracy = 0.0
        precision = 0.0
        recall = 0.0
        f_score = 0.0
        for i in range(full_pm_matrix.shape[0]):
            accuracy += full_pm_matrix[i][j].accuracy
            precision += full_pm_matrix[i][j].precision
            recall += full_pm_matrix[i][j].recall
            f_score += full_pm_matrix[i][j].f_score
        accuracy_per_n.append(accuracy/k)
        precision_per_n.append(precision/k)
        recall_per_n.append(recall/k)
        f_score_per_n.append(f_score/k)

    print(accuracy_per_n)
    print(precision_per_n)
    print(recall_per_n)
    print(f_score_per_n)

    plt.plot(n_vals, accuracy_per_n, marker="o")
    plt.xlabel("n")
    plt.ylabel("Accuracy")
    plt.show()

    plt.plot(n_vals, precision_per_n, marker="o")
    plt.xlabel("n")
    plt.ylabel("Precision")
    plt.show()

    plt.plot(n_vals, recall_per_n, marker="o")
    plt.xlabel("n")
    plt.ylabel("Recall")
    plt.show()

    plt.plot(n_vals, f_score_per_n, marker="o")
    plt.xlabel("n")
    plt.ylabel("F Score")
    plt.show()


def load_dataset(filename, columns_to_drop):
    data = pd.read_csv(filename)
    data = data.drop(columns_to_drop, axis=1)
    data.iloc[:, -1] = data.iloc[:, -1].map({'Y': 1, 'N': 0})

    uniqueval_dict={}
    for column in data.columns[:-1]:
        uniqueval_dict[column] = len(pd.unique(data[column]))
    data2 = EncodeOrNormalizing(data, uniqueval_dict)
    return data2


def EncodeOrNormalizing(data, uniqueval_dict):
    for key, val in uniqueval_dict.items():
        if val > 4:
            # print("normalize")
            normdata = Normalizationdf(data[key])
            data = data.drop(key, axis=1)
            data = pd.concat([data,normdata],axis=1)
            # data=data.drop(key, axis=1)
            # print("data_normalized for-",key, normdata)
        else:
            # print("category :encode ")
            data_encoded = pd.get_dummies(data[key], prefix=key, prefix_sep="_")
            data = pd.concat([data,data_encoded],axis=1)
            data = data.drop(key, axis=1)
            # print("data_encoded for-",key,data_encoded)
    # print("manipulated data:",data.columns.values)
    return data


def Normalizationdf(data):
    data_Norm = (data - data.min()) / (data.max() - data.min())
    return data_Norm


def main():
    filename = 'loan.csv'
    df_type = 'numerical'
    class_column_name = 'Loan_Status'
    columns_to_drop = ['Loan_ID']

    start = time.time()
    dataset = load_dataset(filename, columns_to_drop)
    print(dataset.head)

    k = 10
    n_vals = [1, 5, 10, 20, 30, 40, 50]
    min_size_for_split = round(0.07 * len(dataset.index))

    class_column_idx = dataset.columns.get_loc(class_column_name)
    k_folds = k_fold_split(dataset, k, class_column_idx, class_column_name)
    full_pm_matrix = numpy.empty((k, len(n_vals)), dtype=PerformanceMetrics)

    for i in range(k):
        print("k: ", i)
        test = k_folds[i].sample(frac=1).reset_index(drop=True)
        train = pd.concat([x for a, x in enumerate(k_folds) if a != i]).sample(frac=1).reset_index(drop=True)
        for j in range(len(n_vals)):
            print("n: ", n_vals[j])
            n_trees = []
            bootstraps = create_bootstraps(train, n_vals[j])
            for bootstrap in bootstraps:
                root = Node()
                tree = dev_tree(bootstrap, class_column_idx, root, min_size_for_split, df_type)
                n_trees.append(tree)
            pm = evaluate_random_forest(n_trees, test, class_column_name, df_type)
            full_pm_matrix[i][j] = pm
    end = time.time()
    print("Time taken: ", end-start)
    plot_graph(full_pm_matrix, n_vals, k)


if __name__ == "__main__":
    main()
