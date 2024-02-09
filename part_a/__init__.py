from knn import knn_main
from item_response import item_response_main
from matrix_factorization import matrix_factorization_main
from ensemble import ensemble_main

def student_information ():
    #####################################################################
    # TODO:                                                             #
    # Please complete requested information                             #
    #####################################################################
    information = {
        'First_Name':'Hamed',
        'Last_Name':'Aghaei Karkaj',
        'Student_ID':'401203958',
        'Submission_Date':'16 Khordad 1402',       # In the Persian calendar [Khorshidi] format  
    }
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return information

def test_student_code():
    
    results = {
        'Student_info':student_information(),
        'KNN': knn_main(),
        'item_response':item_response_main(),
        'matrix_factorization':matrix_factorization_main(),
        'ensemble':ensemble_main(),
    }
    
    return results