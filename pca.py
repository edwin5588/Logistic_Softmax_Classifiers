import numpy as np
import matplotlib.pyplot as plt
import data



def pca(x_data):
    '''
        Calculate and Return the pricipal components usiing eigen value decomp
    '''
    print(x_data.shape)
    # mean center the data
    data = x_data - x_data.mean(axis=0)

    # perform eigen value decomp, eigen_vectors are the pricipal components
    covariance_matrix = np.cov(data.T)
    print('Covariance matrix')
    print(covariance_matrix, covariance_matrix.shape)
    print('covariance_matrix == covariance_matrix.T is :',
          (covariance_matrix == covariance_matrix.T).all())
    print('np.iscomplex(covariance_matrix).any() is',
          np.iscomplex(covariance_matrix).any())

    eigen_values, eigen_vectors = np.linalg.eigh(covariance_matrix)
    # sort in acsending orde then reverse indxs
    indxs = eigen_values.argsort()[::-1]
    eigen_vectors = eigen_vectors[:, indxs]
    eigen_values = eigen_values[indxs]

    print("Eigenvector: ", eigen_vectors.shape)
    print("Eigenvalues: ", eigen_values.shape)
    print('First 10 eigen values', eigen_values[:10])
    # calc the explained variance ratio
    var = []
    for val in eigen_values:
        var.append((val / sum(eigen_values)) * 100)
    print('explained variance', var[:10])
   
    # plt.plot([i for i in range(1, eigen_values.size + 1)], np.cumsum(np.array(var)))
    # plt.xlabel("Number of components")
    # plt.ylabel("Cumulative explained variance")
    # plt.title("Explained variance vs Number of components")
    # plt.show()
    return eigen_vectors


def pca_figure1(x_data, y_labels):
    rand_image = x_data[50, :]
    rand_image = np.reshape(rand_image, (1, rand_image.size))
    img = np.reshape(rand_image, (28, 28))
    plt.imshow(img, cmap=plt.get_cmap('gray'))
    plt.show()

    prin_comp = pca(x_data[:10000])
 
    P = [2, 10, 50, 100, 200, 784]
    for p in P:
        # reason for first transpose is to make column ordered then the second transpose
        # is convert back to row ordered
        project = (prin_comp.T[:][:p]).T
        print('project', project.shape)
        reduced = rand_image.dot(project)
        recovered = reduced.dot(project.T)

        img = np.reshape(recovered, (28, 28))
        plt.imshow(img, cmap=plt.get_cmap('gray'))
        plt.show()

def pca_figure2(x_data, y_labels):
    P = 10
    classes = [5, 3, 8]
    indxs1 = np.argwhere(y_labels == 0)
    indxs2 = np.argwhere(y_labels == 9)
    indxs = np.append(indxs1, indxs2)
    # indxs = indxs1[:, 0]
    print(indxs.shape)
    prin_comp = pca(x_data[indxs])

    img = np.reshape(x_data[indxs[0]], (28, 28))
    plt.imshow(img, cmap=plt.get_cmap('gray'))
    plt.show()

    for p in range(P):
        # show a pricipal compomenent
        img = np.reshape(prin_comp[:, p], (28, 28))
        plt.imshow(img, cmap=plt.get_cmap('gray'))
        plt.show()


if __name__ == "__main__":
    # get the data
    from fashion_mnist_dataset.utils import mnist_reader
    X_train, y_train = mnist_reader.load_mnist('fashion_mnist_dataset/data/fashion', kind= 'train')
    X_test, y_test = mnist_reader.load_mnist('fashion_mnist_dataset/data/fashion', kind= 't10k')

    # part 1 and 2 stuff    
    norm_data = data.min_max_norm(X_train)
    x_shuffled, y_shuffled = data.shuffle(norm_data, y_train)

    # part 3 PCA
    pca_figure1(x_shuffled, y_shuffled)
    pca_figure2(x_shuffled, y_shuffled)
