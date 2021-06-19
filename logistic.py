import numpy as np
import matplotlib.pyplot as plt
import andrew
import data
from scipy.special import expit


# helper functions
def get_sets(x, y, set_type):
    # ensure that length of x == y
    '''gets only the sets required for logistic regression'''
    if len(x) != len(y):
        return "the indices are not equal!"
    elif set_type == "set1":
        # if 1, then the category is T-shirt, if not then it's ankleboot
        X_ret, y_ret = x[(y[:, 0] == 1) | (y[:, 9] == 1)], y[(y[:, 0] == 1) | (y[:, 9] == 1)]
        return X_ret, y_ret[:, 0]
    elif set_type == "set2":
        # if 1 then it's pullover, if not it's a coat
        X_ret, y_ret = x[(y[:, 2] == 1) | (y[:, 4] == 1)], y[(y[:, 2] == 1) | (y[:, 4] == 1)]
        return X_ret, y_ret[:, 2]

def sigmoid(x):
    '''sigmoid function'''
    return 1 / (1 + np.exp(-x))

def randomize(length):
    '''shuffles the indices'''
    ind = np.arange(length)
    np.random.shuffle(ind)
    return ind

def loss(output, target):
    '''cross entropy function'''
    out = -np.sum(target * np.log(output + 0.0000000001) + (1 - target) * np.log((1 - output) + 0.0000000001))
    return out


def calc_loss(X, w, y):
    '''calculates the loss using trained weights.'''
    a = np.dot(X, w)
    output = sigmoid(a)

    return loss(output, y)

def fit(X, y, B, lr, w, X_val, y_val):
    '''
    fits the logistic regression
    X --> batched X
    y --> batched y
    B --> batches
    lr --> learning rate
    w --> weight vector of the last iteration
    X_val --> validation set
    y_val --> target for validation
    '''

    outputs = []
    train_err, val_err = [], []
    l = w.copy()

    # randomize the order of the indices
    ind = randomize(len(X))
    X_shuffled, y_shuffled = X[ind], y[ind]

    # B = Batch sizes
    for j in range(0, len(X_shuffled), B):
        start = j
        end = j + B

        X_batch = X_shuffled[start:end]
        target_batch = y_shuffled[start:end]
        # a = X.w
        # g(a) = sigmoid(a)
        a = np.dot(X_batch, l)

        output = sigmoid(a)
        outputs.append(output)
        # l -= lr * np.dot(X_batch.T, (target_batch - output))
        l -= lr * np.dot(X_batch.T, (output - target_batch))

        # calculate losses
        val_loss = calc_loss(X_val, l, y_val)/len(X_val)
        train_loss = calc_loss(X_batch, l, target_batch)/len(X_batch)

        train_err.append(train_loss)
        val_err.append(val_loss)


    return [l, np.mean(train_err), np.mean(val_err)]

def add_bias(X):
    '''
    X --> data to add bias
    Adds a column of 1's depending on the shape of X

    '''
    ones = np.ones((len(X), 1), dtype = int)
    return np.append(ones, X, axis = 1)

def training_procedure(X_train, y_train, set_type, p= 8, batch = 256, lr = 0.01):
    '''
    training procedure according to the Programming assignment
    outputs a list of training losses, validation losses, best validation loss value,
    weight vector based on the best validation value, and the accuracy of the

    X_train --> training set
    y_train --> target labels to train set
    set_type --> set 1 or set 2
    p --> # of principal principal components
    batch --> batch size
    lr --> learning rate

    '''
    norm_data = data.min_max_norm(X_train)
    y_hot = data.one_hot_encode(y_train)
    x_shuffled, y_shuffled = data.shuffle(norm_data, y_hot)
    pcs = andrew.pca(x_shuffled)
    X_sets, y_sets = data.K_validation_datatset(x_shuffled, y_shuffled)
    best_w = np.zeros(p+1)
    best_loss = 100000
    train_losses, val_losses, outputs = [], [], []
    for fold in range(10):
        X_val_set, y_val_set = X_sets[fold], y_sets[fold]
        X = np.concatenate(X_sets[np.arange(10) != fold])
        y = np.concatenate(y_sets[np.arange(10) != fold])
        # project train set and val set onto top p train Principal components
        project = pcs[:, :p]
        reduced_val = np.dot(X_val_set, project)
        reduced_val = add_bias(reduced_val)
        X_val, y_val = get_sets(reduced_val, y_val_set, set_type)
        X = np.dot(X, project)
        X = add_bias(X)
        X, y = get_sets(X, y, set_type)
        #project train_Set and val_set onto top p train PC's
        w = np.zeros(p + 1)
        val_loss = []
        train_loss = []
        early_stop = 0
        last = False
        for epoch in range(100):
            ind = np.arange(len(X))
            np.random.shuffle(ind)
            X, y = X[ind], y[ind]

            # performance on val_set
            w, mean_train_err, mean_val_err = fit(X, y, batch, lr, w, X_val, y_val)


            # early stopping and saving the best model based on val_set performance
            # val_perf = calc_loss(X_val, w, y_val)
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
            # if by 10 epochs the loss is increasing
            elif (early_stop == 10) & (last == True):
                w, mean_train_err, mean_val_err = fit(X, y, batch, lr, best_w, X_val, y_val)
                train_loss.append(mean_train_err)
                val_loss.append(mean_val_err)
            else:
                train_loss.append(mean_train_err)
                val_loss.append(mean_val_err)


        train_losses.append(train_loss)
        val_losses.append(val_loss)
    accuracy = performance(y, np.dot(X, best_w))

    return [np.array(train_losses), np.array(val_losses), best_loss, best_w,accuracy]

def hyperparam_tuning(X_train, y_train,X_test, y_test, set_type):

    '''
    skeleton for hyperparameter tuning. Prints the best loss, best principal component, best batch size, best learning rate on to the console.
    Also prints the accuracy of the best model on the test set.
    '''
    learning_rate = [0.001, 0.01, 0.1]
    batch_size = [512, 256, 128, 64]
    principal_components = [2,10,14,20]

    bestest_loss = 10000
    bestest_w = np.array([])
    best_p = 0
    best_batch_size = 0
    best_learning_rate = 0

    for lr in learning_rate:
        for bs in batch_size:
            for pc in principal_components:
                train_loss, val_loss, best_loss, best_w, acc = training_procedure(X_train, y_train, set_type, pc, bs, lr)
                print("For set: ", set_type, ", learning rate: ", lr, ", batch size: ", bs, ", principal components: ", pc, ", \
                the average loss on the validation set is: ", best_loss)

                create_figure(train_loss, val_loss, [pc, bs, lr, set_type, acc])



                if best_loss < bestest_loss:
                    bestest_loss = best_loss
                    bestest_w = best_w
                    best_p = pc
                    best_batch_size = bs
                    best_learning_rate = lr


    print("Best loss: ", bestest_loss, ", best_p: ", best_p, ", best_BS: ", best_batch_size, ", best_LR: ", best_learning_rate)

    # after hyperparam_tuning, we run the test set
    X_norm_train = data.min_max_norm(X_train)
    X_norm_test = data.min_max_norm(X_test)
    pcs = andrew.pca(X_norm_train)
    project = pcs[:, :best_p]
    X = np.dot(X_norm_test, project)
    y_hot = data.one_hot_encode(y_test)

    print("****************************************************")
    # for set 1
    X, y = get_sets(X, y_hot, set_type)
    X = add_bias(X)
    a = np.dot(X, bestest_w)
    out = sigmoid(a)
    perf = performance(y, out)
    print("For %s, the accuracy of the final model on the test set is: "%set_type, perf)



def create_figure(trainingL, validationL, params):
    '''
    training and validation losses should be of shape (10, 100)
    params should be a list: [pc, batch size, lr, set, accuracy]

    creates the loss curve for training and validation.

    '''
    # ensure that the shape is (10, 100)

    if (trainingL.shape == (10, 100)) & (validationL.shape == (10, 100)):
        plt.plot(np.mean(trainingL, axis = 0), label = 'training loss')
        plt.plot(np.mean(validationL, axis = 0), label = 'validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('Average Cross Entropy Loss')
        plt.legend()
        plt.title("Training V.S. Validation Loss for %s, \n PC:%d, LR:%.4f, Batch:%d, Accuracy:%.4f" %(params[3], params[0], params[2], params[1],params[4]))
        plt.savefig("photos/%s %d PC %d BS %s LR" %(params[3], params[0], params[1], str(params[2]).replace(".", "-")), dpi = 158)
        plt.clf()



def performance(target, output):
    '''returns the percentage performance (accuracy)'''
    output[output > 0.5] = 1
    output[output <= 0.5] = 0

    if len(target) == len(output):
        return np.sum(target == output)/len(target)
    else:
        return "Target and output are different lengths!!"

if __name__ == "__main__":
    # get the data
    from fashion_mnist_dataset.utils import mnist_reader
    X_train, y_train = mnist_reader.load_mnist('fashion_mnist_dataset/data/fashion', kind= 'train')
    X_test, y_test = mnist_reader.load_mnist('fashion_mnist_dataset/data/fashion', kind= 't10k')

    hyperparam_tuning(X_train, y_train, X_test, y_test, "set1")
    hyperparam_tuning(X_train, y_train, X_test, y_test, "set2")
