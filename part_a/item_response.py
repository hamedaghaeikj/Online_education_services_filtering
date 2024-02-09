from utils import load_train_sparse,sparse_matrix_evaluate,load_valid_csv,load_public_test_csv,load_train_csv

import numpy as np
import matplotlib.pyplot as plt
import os 
import random

def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0.
    
    user_ids = data['user_id']
    question_ids = data['question_id']
    is_correct = data['is_correct']

    for user_id, question_id, correct in zip(user_ids, question_ids, is_correct):
        theta_i = theta[user_id]
        beta_j = beta[question_id]

        log_lklihood += correct *np.log(sigmoid(theta_i - beta_j)) + (1-correct) *np.log(1-sigmoid(theta_i - beta_j))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    user_ids = data['user_id']
    question_ids = data['question_id']
    is_correct = data['is_correct']

    for user_id, question_id, correct in zip(user_ids, question_ids, is_correct):
        theta_i = theta[user_id]
        beta_j = beta[question_id]

        # Update theta
        theta[user_id] += lr * (correct - sigmoid(theta_i - beta_j))

        # Update beta
        beta[question_id] -= lr * (correct - sigmoid(theta_i - beta_j))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
        
    user_ids = data["user_id"]
    question_ids = data['question_id']

    val_user_ids = val_data['user_id']
    val_question_ids = val_data['question_id']
    
    num_users = max(max(user_ids), max(val_user_ids)) + 1
    num_questions = max(max(question_ids), max(val_question_ids)) + 1
    theta = np.zeros(num_users)
    beta = np.zeros(num_questions)
    
    val_acc_lst = []
    neg_lld_lst = []


    #####################################################################
    # TODO:Complete the code                                            #
    ##################################################################### 
    for i in range(iterations):
        # Update theta and beta using gradient descent
        theta,beta=update_theta_beta(data, lr, theta, beta)
        
        # Compute negative log-likelihood
        neg_loglikelihood = neg_log_likelihood(data, theta, beta)
        
        # Compute validation accuracy
        val_accuracy =evaluate(val_data, theta, beta)

        # Append validation accuracy and negative log-likelihood to the lists
        val_acc_lst.append(val_accuracy)
        neg_lld_lst.append(neg_loglikelihood)

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################       
    
    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst, neg_lld_lst


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def item_response_main():
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
    # Part B:                                                           #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    # -> Important Note: save plots instead of showing them!            #
    #####################################################################
    num_iterations = None
    lr = None
    best_val_acc = 0
    best_neg = 0
    best_lr = None
    best_num_iterations = None
    print()
    for num_iterations in [5,10,30]:
        for lr in [ 0.01, 0.1, 0.001]:
            # Train the IRT model
            theta, beta, val_acc_lst, neg_lld_lst = irt(train_data, val_data, lr, num_iterations)
            # Compute validation accuracy
            val_accuracy = val_acc_lst[-1]  
            neg = neg_lld_lst[-1]
            # Check if the current hyperparameters result in better validation accuracy
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                best_lr = lr
                best_neg=neg
                best_num_iterations = num_iterations
            print(f'Lr={lr}, iteration={num_iterations}, val acc = {val_accuracy:.4f}')

    # Update the hyperparameters with the best values
    lr = best_lr
    num_iterations = best_num_iterations
    
    theta, beta, val_acc_lst, neg_lld_lst = irt(train_data, val_data, best_lr, best_num_iterations)
    learned_theta:np.array = theta
    learned_beta:np.array = beta
    val_acc_list:list = val_acc_lst
    neg_lld_lst:list = neg_lld_lst
    
    # Plot validation accuracy
    plt.figure(figsize=(12,8))
    plt.plot(val_acc_list)
    plt.xlabel('Iterations')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.savefig(os.path.join(data_path, 'plots\IRT\\validation_accuracy.png'))
    plt.show()
        
    # Plot negative log-likelihood
    plt.figure(figsize=(12,8))
    plt.plot(neg_lld_lst)
    plt.xlabel('Iterations')
    plt.ylabel('Negative Log-Likelihood')
    plt.title('Negative Log-Likelihood')
    plt.savefig(os.path.join(data_path, 'plots\IRT\\negative_log_likelihood.png'))
    plt.show()
    print()
   
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


    #####################################################################
    # Part C:                                                           #
    # Best Results                                                      #
    # -> Important Note: save plots instead of showing them!            #
    #####################################################################
    
    final_validation_acc = val_acc_lst[-1]
    print(f"Hyperparameters: Lr={best_lr}, Iterations={best_num_iterations} \n Validation Accuracy= {best_val_acc:.4f}, best_neg_log={best_neg:.4f}")

    print()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # Part D:                                                           #
    # Plots                                                             #
    # -> Important Note: save plots instead of showing them!            #
    #####################################################################
    num_q= max(max(train_data["question_id"]), max(val_data['question_id'])) + 1
    num_Q= 5
    question_ls = random.choices(np.arange(num_q),k=num_Q)
    theta_r=np.linspace(-4,4,100)

    lis=[]
    prob=[]
    
    plt.figure(figsize=(12,8))
    for i in range(num_Q):
        question=question_ls[i]
        lis.append(str(question))
        prob.append(sigmoid(theta_r-beta[question]*np.ones(theta_r.shape)))
        plt.plot(theta_r,prob[i])
    plt.xlabel('Theta')
    plt.ylabel('Probability')
    plt.legend(lis)
    plt.savefig(os.path.join(data_path, 'plots\IRT\probability.png'))
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


    results = {
        'lr':lr,
        'num_iterations':num_iterations,
        'theta':learned_theta,
        'beta':learned_beta,
        'val_acc_list':val_acc_list,
        'neg_lld_lst':neg_lld_lst,
        'final_validation_acc':final_validation_acc,
        }
    return results



if __name__ == "__main__":
    item_response_main()
