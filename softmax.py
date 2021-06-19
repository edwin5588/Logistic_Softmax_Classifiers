import numpy as np
import matplotlib.pyplot as plt
import data
import andrew
from tqdm import tqdm

def softmax(a):
    # print(a, a.shape)
    if np.isnan(a).any():
        exit()
    return np.exp(a) / np.atleast_2d( np.sum(np.exp(a), axis =1)).T

def evaluate():
    calc_loss()
    return np.argmax(outputs, axis=1)

def loss(target, output):
    e = 1.0e-7
    loss = -np.mean(target * np.log(output + e))
    return loss

def calc_loss(inputs, weights, target):
    a = inputs @ weights
    outputs = softmax(a)
    return loss(target, outputs)
def cross_entropy(X, y, t):
    return -(X.T @ (t - y))

def sgd(weights, outputs, inputs, targets, lr):
    n = len(outputs)

    # generate and randomize the indices
    indices = np.arange(n)
    np.random.shuffle(indices)

    for i in range(n):
        gradients = cross_entropy(
            inputs[indices[i]][np.newaxis, :],
            outputs[indices[i]][np.newaxis, :],
            targets[indices[i]][np.newaxis, :])
        weights = weights - (lr * gradients)

    return weights

def add_bias(X):
    ones = np.ones((len(X), 1), dtype = int)
    return np.append(ones, X, axis = 1)

def randomize(length):
    ind = np.arange(length)
    np.random.shuffle(ind)
    return ind

def concat(X_set, y_set):
    '''
    combines the sets
    '''

    #ensure that the sets have the same lengths
    if len(X_set) == len(y_set):
        X, y = np.array([]), np.array([])
        for ind in range(len(X_set)):
            X = np.concatenate(X, X_set[ind])
            y = np.concatenate(y, y_set[ind])
        return [X, y]
    else:
        return "The lengths of X_set and y_set do not match"

def fit(X, y, B, lr, w, X_val, y_val):
    '''
    fits the logistic regression
    X --> batched X
    y --> batched y
    B --> batches
    lr --> learning rate
    w --> weight vector of the last iteration
    x_val
    '''
    outputs = []
    train_err, val_err = [], []
    l = w.copy()

    # randomize the order of the indices
    ind = randomize(len(X))
    X_shuffled, y_shuffled = X[ind], y[ind]

    # B = Batch sizes
    for j in range(0, len(X), B):
        start = j
        end = j + B

        X_batch = X[start:end]
        target_batch = y[start:end]
        a = X_batch @ l
        output = softmax(a)

        # update weights - stochastic gd
        l = sgd(l, output, X_batch, target_batch, lr)

        # calculate losses
        val_loss = calc_loss(X_val, l, y_val)/len(X_val)
        train_loss = loss(output, target_batch)

        train_err.append(train_loss)
        val_err.append(val_loss)

    return l, np.mean(train_err), np.mean(val_err)

def training_procedure(X_train, y_train, p, batch, lr):
    y_hot = data.one_hot_encode(y_train)
    x_shuffled, y_shuffled = data.shuffle(X_train, y_hot)
    pcs = andrew.pca(x_shuffled)
    X_sets, y_sets = data.K_validation_datatset(x_shuffled, y_shuffled)
    
    best_w = np.zeros((p + 1, 10))
    best_loss = 100000
    train_losses, val_losses, outputs = [], [], []
    
    for fold in tqdm(range(10)):
        X_val_set, y_val_set = X_sets[fold], y_sets[fold]
        X = np.concatenate(X_sets[np.arange(10) != fold])
        y = np.concatenate(y_sets[np.arange(10) != fold])
        # project train set and val set onto top p train Princial components
        project = pcs[:, :p]

        reduced_val = np.dot(X_val_set, project)
        reduced_val = add_bias(reduced_val)
        
        X = np.dot(X, project)
        X = add_bias(X)
        
        w = np.zeros((p + 1, 10))
        val_loss = []
        train_loss = []
        early_stop = 0
        last = False
        for epoch in range(100):
            ind = np.arange(len(X))
            np.random.shuffle(ind)
            X, y = X[ind], y[ind]

            # performance on val_set
            w, mean_train_err, mean_val_err = fit(
                X, y, batch, lr, w, reduced_val, y_val_set)

            # early stopping and saving the best model based on val_set performance
            val_perf = mean_val_err
            if (val_perf < best_loss):
                last = False
                early_stop = 0
                best_loss = val_perf
                best_w = w
                train_loss.append(mean_train_err)
                val_loss.append(mean_val_err)
            elif (val_perf > best_loss):
                last = True
                early_stop += 1
                train_loss.append(mean_train_err)
                val_loss.append(mean_val_err)
            # if by 3 epochs the loss is increasing
            elif (early_stop == 10) & (last == True):
                w, mean_train_err, mean_val_err = fit(X, y, batch, lr, best_w, reduced_val, y_val_set)
                train_loss.append(mean_train_err)
                val_loss.append(mean_val_err)
            else:
                train_loss.append(mean_train_err)
                val_loss.append(mean_val_err)


        train_losses.append(train_loss)
        val_losses.append(val_loss)

    return [np.array(train_losses), np.array(val_losses), best_loss, best_w]

def performance(target, output):
    '''returns the percentage performance (accuracy)'''
    output[output > 0.5] = 1
    output[output <= 0.5] = 0

    if len(target) == len(output):
        return np.sum(target == output)/len(target)
    else:
        return "Target and output are different lengths!!"

def hyperparam_tuning(X_train, y_train, X_test, y_test):

    learning_rate = [0.01]
    batch_size = [256]
    principal_components = [10, 14, 50, 70]

    bestest_loss = 5
    bestest_w = np.array([])
    best_p = 0
    best_batch_size = 0
    best_learning_rate = 0

    for lr in learning_rate:
        for bs in batch_size:
            for pc in principal_components:
                train_loss, val_loss, best_loss, best_w = training_procedure(X_train, y_train, pc, bs, lr)
                print("learning rate: ", lr, ", batch size: ", bs, ", principal components: ", pc, ", \
                the average loss on the validation set is: ", best_loss)

                create_figure(train_loss, val_loss, [pc, bs, lr])

                if best_loss < bestest_loss:
                    bestest_loss = best_loss
                    bestest_w = best_w
                    best_p = pc
                    best_batch_size = bs
                    best_learning_rate = lr


    return bestest_loss, bestest_w, best_p, best_batch_size, best_learning_rate


def create_figure(trainingL, validationL, params):
    '''
    training and validation losses should be of shape (10, 100)
    params should be a list: [pc, batch size, lr, set]
    '''
    # ensure that the shape os (10, 100)

    if (trainingL.shape == (10, 100)) & (validationL.shape == (10, 100)):
        plt.plot(np.mean(trainingL, axis = 0), label = 'training loss')
        plt.plot(np.mean(validationL, axis = 0), label = 'validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('Average Loss')
        plt.legend()
        plt.title("Training V.S. Validation Loss: pc= "+str(params[0])+", batch size= "+str(params[1])+", learning rate= "+str(params[2]))
        plt.savefig("photos3/%d PC %d BS %s LR" %(params[0], params[1], str(params[2]).replace(".", "-")), dpi = 158)
        plt.clf()


if __name__ == "__main__":
    # get the data
    from fashion_mnist_dataset.utils import mnist_reader
    X_train, y_train = mnist_reader.load_mnist(
        'fashion_mnist_dataset/data/fashion', kind='train')
    X_test, y_test = mnist_reader.load_mnist(
        'fashion_mnist_dataset/data/fashion', kind='t10k')
    
    X_train = data.min_max_norm(X_train)
    X_test = data.min_max_norm(X_test)
    
    bestest_loss, bestest_w, best_p, best_batch_size, best_learning_rate = hyperparam_tuning(X_train, y_train, X_test, y_test)

    print("best loss: ", bestest_loss)
    print("best w: ", bestest_w)
    print("best p: ", best_p)
    print("best batch size: ", best_batch_size)
    print("best_learning_rate: ", best_learning_rate)







