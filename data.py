import numpy as np
import matplotlib.pyplot as plt



def min_max_norm(data):
    '''
        The formula is: x_scaled = (x - min_val) / (max_val - min_val)

        According to discussion the min_val and max_val is across the entire dataset.
    '''
    max_val = np.max(data)
    min_val = np.min(data)

    def calc(val, min_val, max_val):
        return (val - min_val) / (max_val - min_val)

    norm_func = np.vectorize(calc)
    return norm_func(data, min_val, max_val)


def one_hot_encode(labels):
    '''
        There are 10 classes, so each index on the y label is length 10 
        of 1 hot encoded classes
    '''
    shape = (labels.size, labels.max()+1)
    one_hot = np.zeros(shape)
    rows = np.arange(labels.size)
    one_hot[rows, labels] = 1
    return one_hot

def shuffle(x_data, y_labels):
    new_idxs = np.random.permutation(len(x_data))
    x_shuffled = x_data[new_idxs]
    y_shuffled = y_labels[new_idxs]
    return x_shuffled, y_shuffled



def K_validation_datatset(x_data, y_labels):
    '''
        https://machinelearningmastery.com/k-fold-cross-validation/

        Shuffle the data first to get an somewhat even distribution
        and split the data into k groups
    '''
    if x_data.shape[0] != y_labels.shape[0]:
        return Exception('Not the same size!!')

    K = 10
    size = int(y_labels.shape[0] / K)
    x_sets = np.empty((K, size, x_data.shape[1]))
    y_sets = np.empty((K, size, y_labels.shape[1]))
    # print(x_data.shape, y_labels.shape)
    start = 0
    end = size
    for k in range(K):
        # print(start, end)
        x_sets[k, ...] = x_data[start:end, :] 
        y_sets[k, ...] = y_labels[start:end, :]
        start = end
        end += size
    print('K_validation_datatset')
    print('X shape: ', x_sets.shape)
    print('Y shape: ', y_sets.shape)
    return x_sets, y_sets


if __name__ == "__main__":
    # get the data
    from fashion_mnist_dataset.utils import mnist_reader
    X_train, y_train = mnist_reader.load_mnist('fashion_mnist_dataset/data/fashion', kind= 'train')
    X_test, y_test = mnist_reader.load_mnist('fashion_mnist_dataset/data/fashion', kind= 't10k')
    print('Test')
    print('Images are 28 X 28, but are given as vector of len 784')
    print(X_train, X_train.shape, type(X_train))
    print(y_train, y_train.shape)
    print('Test')
    print(X_test, X_test.shape)
    print(y_test, y_test.shape)

    print('Show an example')
    img = np.reshape(X_train[5], (28, 28))
    plt.imshow(img, cmap=plt.get_cmap('gray'))
    plt.show()

    print('Test Norm')
    test = np.array([2, 1, 0.5, 10])
    print(test)
    norm_data = min_max_norm(test)
    print(norm_data)


    print('Test one hot encoding')
    print(y_test[0])
    print(one_hot_encode(y_test)[0])


    print('Test Shuffle')
    arr1 = np.random.rand(10)
    arr2 = np.random.rand(10)
    print(arr1)
    print(arr2)
    arr1, arr2 = shuffle(arr1, arr2)
    print(arr1)
    print(arr2)

    print('Test Cross Validation Procedure')
    y_hot = one_hot_encode(y_train)
    x_shuffled, y_shuffled = shuffle(X_train, y_hot)
    K_validation_datatset(x_shuffled, y_shuffled)
