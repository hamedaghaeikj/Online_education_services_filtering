from sklearn.impute import KNNImputer
from utils import load_train_sparse,sparse_matrix_evaluate,load_valid_csv,load_train_csv,load_public_test_csv
#####################################################################
# TODO:                                                             #
# Import packages you need here                                     #
#####################################################################
import matplotlib.pyplot as plt
import os
#####################################################################
#                       END OF YOUR CODE                            #
#####################################################################  

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
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

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
    completed_mat = imputer.fit_transform(matrix.T)  
    acc = sparse_matrix_evaluate(valid_data, completed_mat.T)

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def knn_main():
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

    #####################################################################
    # Part B&C:                                                         #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*. do all these using knn_impute_by_user().                                                       #
    #####################################################################

    user_best_k:float = None                    # :float means that this variable should be a float number
    user_test_acc:float = None
    user_valid_acc:list = []
    best_acc=0
    
    K=[1,6,11,16,21,26]
    print("Accuracy for knn_impute_by_user:")
    for k in K:
        accu=knn_impute_by_user(sparse_matrix, val_data, k)
        user_valid_acc.append(accu)

        if accu>best_acc:
            best_acc=accu
            user_best_k=k
    print(f"Best K for knn_impute_by_user : {user_best_k} ")
    print(f"Best Validation Accuracy for knn_impute_by_user : {best_acc:.4f} ")
    try:
        user_test_acc=knn_impute_by_user(sparse_matrix, test_data, user_best_k)
        print(f"Best Test Accuracy for knn_impute_by_user : {user_test_acc:.4f} ")
    except:
        pass
    
    print("****************************************")   
    plt.figure(figsize=(12 ,8))
    plt.plot(K,user_valid_acc)
    plt.xlabel("K")
    plt.ylabel("Accuracy")
    plt.title("User Val_Acc")
    plt.savefig(os.path.join(data_path, r"plots\knn\user_knn.png"))
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # Part D:                                                           #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*. do all these using knn_impute_by_item().                                                        #
    #####################################################################
    
    question_best_k:float = None
    question_test_acc:float = None
    question_valid_acc:list = []
    
    best_acc=0
    print("Accuracy for knn_impute_by_item:")
    for k in K:
        accu=knn_impute_by_item(sparse_matrix, val_data, k)
        question_valid_acc.append(accu)
        if accu>best_acc:
            best_acc=accu
            question_best_k=k
    print(f"Best K for knn_impute_by_item : {question_best_k} ")
    print(f"Best Validation Accuracy for knn_impute_by_item : {best_acc:.4f} ")
    try:
        question_test_acc=knn_impute_by_item(sparse_matrix, test_data, question_best_k)
        print(f"Best Test Accuracy for knn_impute_by_item : {question_test_acc:.4f} ")
    except:
        pass
        
    plt.figure(figsize=(12,8))
    plt.plot(K,question_valid_acc)
    plt.xlabel("K")
    plt.ylabel("Accuracy")
    plt.title("Question Val_Acc")
    plt.savefig(os.path.join(data_path, "plots\knn\question_knn.png"))
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    results = {
    'user_best_k':user_best_k,
    'user_test_acc':user_test_acc,
    'user_valid_accs':user_valid_acc,
    'question_best_k':question_best_k,
    'question_test_acc':question_test_acc,
    'question_valid_acc':question_valid_acc,
    }
    
    
    return results


if __name__ == "__main__":
    knn_main()
