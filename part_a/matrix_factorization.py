from utils import load_train_sparse,sparse_matrix_evaluate,load_valid_csv,load_public_test_csv,load_train_csv
import os
from scipy.linalg import sqrtm
import numpy as np
import matplotlib.pyplot as plt

def svd_reconstruct(matrix, k):
    """ Given the matrix, perform singular value decomposition
    to reconstruct the matrix.

    :param matrix: 2D sparse matrix
    :param k: int
    :return: 2D matrix
    """
    
    
    #####################################################################
    # TODO:                                                             #
    # Part A:                                                           #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # First, need to fill in the missing values (NaN) to perform SVD.

    column_means = np.nanmean(matrix, axis=0)

    # Fill in missing values with column means
    matrix_filled = np.where(np.isnan(matrix), column_means, matrix)

    # Perform SVD
    U, S, V = np.linalg.svd(matrix_filled)

    # Reconstruct the matrix
    reconst_matrix = np.dot(U[:, :k] * S[:k], V[:k, :])

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return np.array(reconst_matrix)


def squared_error_loss(data, u, z):
    """ Return the squared-error-loss given the data.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param u: 2D matrix
    :param z: 2D matrix
    :return: float
    """
    loss = 0
    for i, q in enumerate(data["question_id"]):
        loss += (data["is_correct"][i]
                 - np.sum(u[data["user_id"][i]] * z[q])) ** 2.
    return 0.5 * loss


def update_u_z(train_data, lr, u, z):
    """ Return the updated U and Z after applying
    stochastic gradient descent for matrix completion.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param u: 2D matrix
    :param z: 2D matrix
    :return: (u, z)
    """
    #####################################################################
    # TODO:                                                             #
    # Part C:                                                           #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # Randomly select a pair (user_id, question_id).
    i = \
        np.random.choice(len(train_data["question_id"]), 1)[0]

   # Randomly select a pair (user_id, question_id).

    c = train_data["is_correct"][i]
    n = train_data["user_id"][i]
    q = train_data["question_id"][i]

    # Compute the error
    error = c - np.dot(u[n, :], z[q, :])

    # Update u and z matrices
    u[n, :] += lr * error * z[q, :]
    z[q, :] += lr * error * u[n, :]

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return u, z


def als(train_data, val_data,k, lr, num_iteration):
    """ Performs ALS algorithm. Return reconstructed matrix.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :param lr: float
    :param num_iteration: int
    :return: 2D reconstructed Matrix.
    """
    # Initialize u and z
    u = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["user_id"])), k))
    z = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["question_id"])), k))

    #####################################################################
    # TODO:                                                             #
    # Part C:                                                           #
    # Implement the function as described in the docstring.             #
    #####################################################################
    squared=[]
    for i in range(num_iteration):
        for j in range(len(set(train_data["question_id"]))):
            u, z = update_u_z(train_data, lr, u, z)
        squared.append(squared_error_loss(val_data,u,z))
    # Reconstruct the matrix
    mat = np.dot(u, z.T)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return mat,squared


def matrix_factorization_main():
    data_path = os.path.abspath(os.curdir)
    os.chdir("..")
    data_path = os.path.abspath(os.curdir)

    train_matrix = load_train_sparse(os.path.join(data_path, "data")).toarray()
    train_data = load_train_csv(os.path.join(data_path, "data"))
    val_data = load_valid_csv(os.path.join(data_path, "data"))
    try:
        test_data = load_public_test_csv(os.path.join(data_path, "data"))
    except:
        print("There is no test_data.csv file")
    #####################################################################
    # TODO:                                                             #
    # Part A:                                                           #
    # (SVD) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################
    
    best_k_svd = 0
    best_val_acc_svd = 0
    test_acc_svd = 0

    # Try out different values of k
    k_values = [1,5,10,15,20,25,30,40]
    val_accs_svd = []
    best_val_acc_svd = 0
    for k in k_values:
        # Perform SVD and reconstruct the matrix
        reconstructed_matrix = svd_reconstruct(train_matrix, k)

        # Compute accuracy on validation set
        val_acc = sparse_matrix_evaluate(val_data,reconstructed_matrix)
        val_accs_svd.append(val_acc)
    
        # Update best k and validation accuracy
        if val_acc >= best_val_acc_svd:
            best_k_svd = k
            best_val_acc_svd = val_acc
    try:
        test_acc_svd = sparse_matrix_evaluate( test_data,reconstructed_matrix)
        print(f'best svd k={best_k_svd} ,best test svd accuracy={test_acc_svd:.4f}')
    except:
        pass
    print(f'best svd k={best_k_svd} ,best svd accuracy={best_val_acc_svd:.4f}')
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Part D and E:                                                     #
    # (ALS) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################
    best_k_als = 0
    best_val_acc_als = 0

    # Perform ALS with different values of k
    k_values_als = [1,5,10,20,30,40]
    Lr=[0.1,0.01,0.001]
    iterations=[30,120]
    val_accs_als = []
    for num_iteration in iterations:
        for lr in Lr:
            for k in k_values_als:
                # Perform ALS and reconstruct the matrix
                reconstructed_matrix_als,squared = als(train_data, val_data,k, lr, num_iteration)
                # Compute accuracy on validation set
                val_acc_als = sparse_matrix_evaluate(val_data,reconstructed_matrix_als )
               
                # Update best k and validation accuracy
                if  val_acc_als > best_val_acc_als:
                    best_k_als = k
                    best_lr=lr
                    best_it=num_iteration
                    best_val_acc_als = val_acc_als
    
    reconstructed_matrix_als,squared = als(train_data, val_data,best_k_als, best_lr, best_it)
    print(f'best k = {best_k_als}, best lr = {best_lr}, best num_iteration = {best_it} \nbest val accuracy = {best_val_acc_als:.4f}')
    print("Reconstructed_matrix_als :")
    print(reconstructed_matrix_als)
    print()

    
    plt.figure(figsize=(12,8))
    plt.plot(squared)
    plt.xlabel('Iterations')
    plt.ylabel('squared_error_loss')
    plt.savefig(os.path.join(data_path, "plots\matrix_factorization\squared_error_loss.png"))
    plt.show()
    
    for k in k_values_als:
        # Perform ALS and reconstruct the matrix
        reconstructed_matrix_als,squred= als(train_data,val_data, k, best_lr, best_it)

        # Compute accuracy on validation set
        val_acc_als = sparse_matrix_evaluate(val_data,reconstructed_matrix_als )
        val_accs_als.append(val_acc_als)
    
    plt.figure(figsize=(12,8))
    plt.plot(k_values, val_accs_svd)
    plt.plot(k_values_als, val_accs_als)
    plt.xlabel('k')
    plt.ylabel('Validation Accuracy')
    plt.legend(['SVD','ALS'])
    plt.savefig(os.path.join(data_path, "plots\matrix_factorization\Validation_vs_k.png"))
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    results={
    'best_k_svd':best_k_svd,
    'test_acc_svd':test_acc_svd,
    'best_val_acc_svd':best_val_acc_svd,
    'best_val_acc_als':best_val_acc_als,
    'best_k_als':best_k_als

    }

    return results

if __name__ == "__main__":
    matrix_factorization_main()
