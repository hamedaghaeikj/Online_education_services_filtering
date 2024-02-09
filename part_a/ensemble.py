import numpy as np
#####################################################################
# TODO:                                                             #                                                          
# Import packages you need here                                     #
#####################################################################
import os
from sklearn.impute import KNNImputer
from utils import load_train_sparse,load_train_csv,sparse_matrix_evaluate,load_valid_csv,load_public_test_csv

#####################################################################
#                       END OF YOUR CODE                            #
#####################################################################  




#####################################################################
# Define and implement functions here                               #
#####################################################################



def bagging(X, num_models):
    """Apply bagging to create an ensemble of models.

    :param X: 2D array-like, feature matrix
    :param num_models: int, number of models to create
    """
    num_samples = len(X)
    Sample=[]
    for _ in range(num_models):
        indices = np.random.choice(num_samples, size=(num_samples,))
        indices = np.random.choice(num_samples, size=(num_samples,))
        X_sampled = X[indices]
        Sample.append(X_sampled)
    return Sample

def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    imputer = KNNImputer(n_neighbors=k)
    completed_mat = imputer.fit_transform(matrix)    
    acc = sparse_matrix_evaluate(valid_data, completed_mat)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################    
    return completed_mat,acc



#####################################################################
#                       END OF YOUR CODE                            #
##################################################################### 




def ensemble_main():
    #####################################################################
    # Compute the finall validation and test accuracy                   #
    #####################################################################
    
    val_acc_ensemble:float = None
    test_acc_ensemble:float = None
    method1_output_matrix:np.array = None
    method2_output_matrix:np.array = None
    method3_output_matrix:np.array = None
    
    data_path = os.path.abspath(os.curdir)
    os.chdir("..")
    data_path = os.path.abspath(os.curdir)

    sparse_matrix = load_train_sparse(os.path.join(data_path, "data")).toarray()
    train_data = load_train_csv(os.path.join(data_path, "data"))
    val_data = load_valid_csv(os.path.join(data_path, "data"))
    try:
        test_data = load_public_test_csv(os.path.join(data_path, "data"))
    except:
        print("There is no test_data.csv file")
    
    num_models=3
    
    Sample=bagging(sparse_matrix, num_models)
    best_k=0
    best_acc=0
    acc=[] 
    k=[1,6,11,16,21,26]
    for K in k:
        method1_output_matrix,accu=knn_impute_by_user(Sample[0], val_data, K)
        method2_output_matrix,accu=knn_impute_by_user(Sample[1], val_data, K)
        method3_output_matrix,accu=knn_impute_by_user(Sample[2], val_data, K)
        matrix= (method1_output_matrix+method2_output_matrix+method3_output_matrix)/3
    
        val_acc_ensemble = sparse_matrix_evaluate(val_data, matrix)
        acc.append(val_acc_ensemble)
        if val_acc_ensemble>best_acc:
            best_acc=val_acc_ensemble
            best_k=K
    
    val_acc_ensemble=best_acc
    method1_output_matrix,accu=knn_impute_by_user(Sample[0], val_data, best_k)
    method2_output_matrix,accu=knn_impute_by_user(Sample[1], val_data, best_k)
    method3_output_matrix,accu=knn_impute_by_user(Sample[2], val_data, best_k)
    matrix= (method1_output_matrix+method2_output_matrix+method3_output_matrix)/3
    print(f"validation Accuracy = {val_acc_ensemble:.4f}")
    
    try:
        test_acc_ensemble= sparse_matrix_evaluate(test_data, matrix)
        print(f"Test Accuracy = {test_acc_ensemble:.4f}")
    except:
        pass
        
    

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    results={
    'val_acc_ensemble':val_acc_ensemble,
    'test_acc_ensemble':test_acc_ensemble,
    'method1_output_matrix':method1_output_matrix,
    'method2_output_matrix':method2_output_matrix,
    'method3_output_matrix':method3_output_matrix
    }

    return results


if __name__ == "__main__":
    ensemble_main()

