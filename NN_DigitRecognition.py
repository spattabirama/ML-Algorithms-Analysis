import numpy
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn import datasets
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score


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


def graph_plot(y, x_range, x_len):
    plt.plot(list(range(0, x_range*x_len, x_len)), y)
    plt.xlabel('input')
    plt.ylabel('cost')
    plt.show()


def initialize_theta(neurons_per_layer):
    theta = []
    for i in range(len(neurons_per_layer)-1):
        shape = (neurons_per_layer[i+1], neurons_per_layer[i]+1)
        t = np.round(np.random.uniform(low=-1, high=1, size=shape), 1)
        theta.append(t)
    theta = [np.atleast_2d(item) for item in theta]
    return theta


def add_bias(arr):
    bias_col = np.ones((arr.shape[0], 1))
    return np.concatenate((bias_col, arr), axis=1)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def fwd_propogation(x, layers_count, theta):
    a_list = []
    a = add_bias(x)
    a_list.append(a)
    #print("a1: ", a)
    #print(f"theta1: ", theta[0])
    for i in range(0, layers_count-2):
        z = a @ theta[i].T
        print(f"z{i+2}: ", z)
        a = sigmoid(z)
        a = add_bias(a)
        a_list.append(a)
        #print(f"a{i+2}: ", a)
        #print(f"theta{i+2}: ", theta[i+1])
    z = a @ theta[layers_count-2].T
    #print(f"z{layers_count}: ", z)
    f = sigmoid(z)
    a_list.append(f)
    return a_list


def one_hot_encode(arr, unique_vals):
    unique_vals = sorted(unique_vals)
    if isinstance(arr[0], (numpy.int64, float)):
        encoded_arr = np.eye(len(unique_vals), dtype=float)[np.searchsorted(unique_vals, arr)]
    else:
        raise ValueError("Unsupported data type")

    return encoded_arr


def preprocess_instances(x_arr, y_arr, cls_unique):
    x = np.expand_dims(x_arr, axis=1)
    y = np.expand_dims(one_hot_encode(y_arr, cls_unique), axis=1)
    return x, y


def calculate_gradients(a_list, y, layer_count, theta):
    delta_dic = {}
    gradient_dic = {}
    delta_dic[layer_count] = (a_list[-1] - y).T
    #print(f"delta[{layer_count}]: ", delta_dic[layer_count])
    for i in range(layer_count-2, 0, -1):
        delta = (theta[i].T @ delta_dic[i+2]).T * a_list[i] * (1 - a_list[i])
        delta_dic[i+1] = delta[:, 1:].T
        #print(f"delta[{i+1}]: ", delta_dic[i+1])
    for i in range(layer_count-2, -1, -1):
        gradient_dic[i+1] = a_list[i] * delta_dic[i+2]
        #print(f"gradient[{i+1}]: ", gradient_dic[i+1])
    return gradient_dic


def calculate_regulated_cost(x, j, theta, lamda):
    s = 0.0
    for t in theta:
        if t.ndim == 1:
            s = np.sum(np.square(t[1:]))
        else:
            s = np.sum(np.square(t[:, 1:]))
    s = (lamda / (2 * len(x))) * s
    cost = (j + s) / len(x)
    return cost


def calculate_regulated_gradients(x, sum_gradient_dic, theta, lamda, layer_count):
    reg_gradient_list = []
    sum_gradient_arr = list(sum_gradient_dic.values())[::-1]
    for i in range(0, layer_count - 1):
        p = lamda * theta[i]
        p[:, 0] = 0
        reg_d = (sum_gradient_arr[i] + p) / len(x)
        #print(f"Regularized gradient of theta{i + 1}:", reg_d)
        reg_gradient_list.append(reg_d)
    return reg_gradient_list


def update_theta(theta, reg_gradient_list, alpha, layer_count):
    for i in range(0, layer_count - 1):
        theta[i] = theta[i] - alpha * reg_gradient_list[i]
    return theta


def back_propogation(x, y, test_x, test_y, cls_unique, neurons_per_layer, theta, lamda, alpha, k_stop):
    x, y = preprocess_instances(x, y, cls_unique)
    j = 0.0
    layer_count = len(neurons_per_layer)
    sum_gradient_dic = {}
    j_per_iter_for_test = []

    for k in range(k_stop):
        for i in range(len(x)):
            a_list = fwd_propogation(x[i], layer_count, theta)
            #print("f: ", a_list[-1])
            #print("y: ", y[i])
            gradient_dic = calculate_gradients(a_list, y[i], layer_count, theta)
            if not sum_gradient_dic:
                sum_gradient_dic = gradient_dic
            else:
                for key in sum_gradient_dic:
                    sum_gradient_dic[key] += gradient_dic[key]
            j_x = 0
            if not np.isin(1.0, a_list[-1]):
                j_x = np.sum(-y[i] * np.log(a_list[-1]) - (1 - y[i]) * np.log(1 - a_list[-1]))
            #print(f"j for {i}th instance:", j_x)
            j += j_x
        reg_cost = calculate_regulated_cost(x, j, theta, lamda)
        reg_gradient_list = calculate_regulated_gradients(x, sum_gradient_dic, theta, lamda, layer_count)
        theta = update_theta(theta, reg_gradient_list, alpha, layer_count)
        if test_x is not None and test_y is not None:
            test_x_p, test_y_p = preprocess_instances(test_x, test_y, cls_unique)
            test_f = np.zeros_like(test_y_p)
            test_j = 0
            for i in range(len(test_x_p)):
                test_f[i] = fwd_propogation(test_x_p[i], len(neurons_per_layer), theta)[-1]
                j_x = 0
                if not np.isin(1.0, test_f[i]):
                    j_x = np.sum(-test_y_p[i] * np.log(test_f[i]) - (1 - test_y_p[i]) * np.log(1 - test_f[i]))
                test_j += j_x
            test_j /= len(test_x_p)
            j_per_iter_for_test.append(test_j)
    if j_per_iter_for_test:
        graph_plot(j_per_iter_for_test, k_stop, len(test_x))
    return theta


def encode_output(output_list):
    encode_list_out = []
    for output in output_list:
        t = np.argmax(output[0])
        z = np.zeros((1, len(output[0])))
        z[0][t] = 1
        encode_list_out.append(z)
    return encode_list_out


def evaluate_neural_network(x, y, cls_unique, neurons_per_layer, theta):
    x, y = preprocess_instances(x, y, cls_unique)
    f = np.zeros_like(y)
    for i in range(len(x)):
        f[i] = fwd_propogation(x[i], len(neurons_per_layer), theta)[-1]
    encoded_f = encode_output(f)
    y = np.squeeze(y)
    encoded_f = np.squeeze(encoded_f)
    decoded_y = np.argmax(y, axis=1)
    decoded_f = np.argmax(encoded_f, axis=1)
    accuracy = (np.sum(decoded_f == decoded_y) / decoded_y.size) * 100
    f_score = f1_score(decoded_y, decoded_f, average='macro') * 100
    return accuracy, f_score


def main():
    start = time.time()

    x, y = load_dataset()
    k_folds = 10
    train_indices, test_indices = k_fold_split(x, y, k_folds)

    n_cls = len(np.unique(y))
    cls_unique = np.unique(y)
    n_att = x.shape[1]
    neurons_per_layer = [n_att, 64, n_cls]
    print(neurons_per_layer)
    theta = initialize_theta(neurons_per_layer)
    lamda = 0.5
    alpha = 0.1
    k_stop_criterion = 1
    accuracy_list = []
    fscore_list = []

    for fold in range(1):
        print("k_fold: ", fold)
        x_train, x_test = x[train_indices[fold]], x[test_indices[fold]]
        y_train, y_test = y[train_indices[fold]], y[test_indices[fold]]
        x_train_min = x_train.min()
        x_train_max = x_train.max()
        x_train = normalize_features(x_train, x_train_min, x_train_max)
        x_test = normalize_features(x_test, x_train_min, x_train_max)

        updated_theta = back_propogation(x_train, y_train, None, None, cls_unique, neurons_per_layer, theta, lamda, alpha, k_stop_criterion)
        accuracy, f_score = evaluate_neural_network(x_test, y_test, cls_unique, neurons_per_layer, updated_theta)
        print("accuracy: ", accuracy)
        print("f_score: ", f_score)
        accuracy_list.append(accuracy)
        fscore_list.append(f_score)

    print("Accuracy of the alg: ", np.average(accuracy_list))
    print("Fscore of the alg: ", np.average(fscore_list))

    end = time.time()
    print("Time taken:", end - start)


def cost_verification():
    k_folds = 5
    x, y = load_dataset()
    train_indices, test_indices = k_fold_split(x, y, k_folds)

    n_cls = len(np.unique(y))
    cls_unique = np.unique(y)
    n_att = x.shape[1]
    neurons_per_layer = [n_att, 64, n_cls]
    print(neurons_per_layer)
    theta = initialize_theta(neurons_per_layer)
    lamda = 0.5
    alpha = 0.1
    k_stop_criterion = 500

    x_train, x_test = x[train_indices[1]], x[test_indices[1]]
    y_train, y_test = y[train_indices[1]], y[test_indices[1]]
    x_train_min = x_train.min()
    x_train_max = x_train.max()
    x_train = normalize_features(x_train, x_train_min, x_train_max)
    x_test = normalize_features(x_test, x_train_min, x_train_max)
    updated_theta = back_propogation(x_train, y_train, x_test, y_test, cls_unique, neurons_per_layer, theta, lamda, alpha, k_stop_criterion)


if __name__ == '__main__':
    # Uncomment main() to run main datasets
    main()
    # cost_verification()
