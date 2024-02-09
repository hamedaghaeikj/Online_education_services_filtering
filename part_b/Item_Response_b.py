from utils import load_train_sparse,sparse_matrix_evaluate,load_valid_csv,load_public_test_csv,load_train_csv
import numpy as np
import matplotlib.pyplot as plt
import os


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta,P, k):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    log_lklihood = 0
    
    user_ids = data['user_id']
    question_ids = data['question_id']
    is_correct = data['is_correct']


    for user_id, question_id, correct in zip(user_ids, question_ids, is_correct):
        theta_i = theta[user_id]
        beta_j = beta[question_id]
        K=k[question_id]
        p=sigmoid((theta_i - beta_j)*K)*(1-P)+P
        log_lklihood += correct *np.log(p) + (1-correct) *np.log(1-p)
    
    return -log_lklihood


def update_parameters( data, lr, theta, beta, P, k, labda):
    """ Update theta, beta , c and k using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    user_ids = data['user_id']
    question_ids = data['question_id']
    is_correct = data['is_correct']
    eps=0.01
    for user_id, question_id, correct in zip(user_ids, question_ids, is_correct):
        theta_i = theta[user_id]
        beta_j = beta[question_id]
        k_j = k[question_id]
        sig=sigmoid((theta_i - beta_j)*k_j)
        p=sig*(1-P)+P
        # Update theta
        theta[user_id] += lr * ((((1-P)*k_j*sig*(1-sig)/(p) + k_j*sig)) * correct - k_j*sig - 2*labda*theta_i)

        # Update beta
        beta[question_id] += lr * (((1-P)*(-k_j)*sig*(1-sig)/(p) - k_j*sig) * correct + k_j*sig - 2*labda*beta_j)
        
        # Update k
        k[question_id] -=  lr * (((1-P)*(theta_i - beta_j)*sig*(1-sig)/(p) -(theta_i - beta_j)*sig) * correct - (theta_i - beta_j)*sig - 2*labda*k_j)
        
        # Update P

        P+= lr*(correct * (1-sig)/(p) - (1-correct)/(1-P))
        P = max(0, P)
        P = min(1-eps, P) # P can't be 1
    return theta, beta, P, k


def irt(data, train_matrix, val_data, lr, iterations, labda):
    """ Train IRT model.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: tuple of vectors
    """
    num_St,num_Qu=train_matrix.shape
    theta = np.ones(num_St) *0.5
    beta = np.ones(num_Qu) *0.5
    k = np.ones((num_Qu,)) 
    P=0.01
    # training and validation negtive log likelihoods
    train_neg_llds, val_neg_llds = [], []
    # training and validation accuracies
    train_accs, val_accs = [], []

    for i in range(iterations):

        train_neg_llds.append(neg_log_likelihood(data, theta, beta, P, k))
        val_neg_llds.append(neg_log_likelihood(val_data, theta, beta, P, k))

        theta, beta, P, k = update_parameters(data, lr, theta, beta, P, k, labda)
        

    return theta, beta, P,  k


def evaluate(data, theta, beta, P, k):
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
        x = ((theta[u] - beta[q]) * k[q]).sum()
        p_a = sigmoid(x)*(1-P)+P
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    
    data_path = os.path.abspath(os.curdir)
    os.chdir("..")
    data_path = os.path.abspath(os.curdir)
    
    data = load_train_csv(os.path.join(data_path, "data"))
    
    train_matrix = load_train_sparse(os.path.join(data_path, "data")).toarray()

    val_data = load_valid_csv(os.path.join(data_path, "data"))
    try:
        test_data = load_public_test_csv(os.path.join(data_path, "data"))
    except:
        print("There is no test_data.csv file")
    
    best_lr=None
    best_lambd=None
    best_it=None
    best_val=0
    for lr in [0.01,0.1]:
        for labda in [0,0.1,0.01]:
            for iterations in [5,10,30]:
                theta, beta, P, k = irt(data, train_matrix, val_data, lr, iterations, labda)
                val_acc=evaluate(val_data, theta, beta, P,  k)
                
                if val_acc>best_val:
                    best_val=val_acc
                    best_it=iterations
                    best_lr=lr
                    best_lambd=labda
    
    theta, beta, P, k = irt(data, train_matrix, val_data, best_lr, best_it, best_lambd)
    print()

    val_acc = evaluate(val_data, theta, beta, P,  k)
    print(f"Hyperparameters: Lr={best_lr}, Iterations={best_it} \nLambda={best_lambd}, Validation Accuracy= {val_acc:.4f}")
    try:
        test_acc = evaluate(test_data, theta, beta, P, k)
        print("test accuracy is {test_acc:.4f}")
    except:
        pass
    



if __name__ == "__main__":
    main()
