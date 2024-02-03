from sklearn.impute import KNNImputer
from utils import *
import matplotlib.pyplot as plt


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
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
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
    
    nbrs = KNNImputer(n_neighbors=k)
    mat = nbrs.fit_transform(matrix.T)
    acc = sparse_matrix_evaluate(valid_data, mat.T)
    
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    
    k_s = [1,6,11,16,21,26]; accuracies_user = []; accuracies_item = []
    
    print("Validation accuracies for kNN imputed by user:")
    for k in k_s:
        a = knn_impute_by_user(sparse_matrix, val_data, k)
        accuracies_user.append(a)
        print(f"k = {k}, validation accuracy = {a}")
    
    print("Validation accuracies for kNN imputed by item:")
    for k in k_s:
        a = knn_impute_by_item(sparse_matrix, val_data, k)
        accuracies_item.append(a)
        print(f"k = {k}, validation accuracy = {a}")
    
    plt.plot(k_s, accuracies_user, label="By user"); plt.plot(k_s, accuracies_item, label="By item"); plt.legend()
    plt.title("Value of k vs accuracy of kNN imputed"); plt.xlabel("Value of k"); plt.ylabel("Accuracy of kNN imputed")
    
    k_user = 11
    print(f"Final test accuracy for kNN imputed by user: {knn_impute_by_user(sparse_matrix, test_data, k_user)}")
    
    k_item = 21
    print(f"Final test accuracy for kNN imputed by item: {knn_impute_by_item(sparse_matrix, test_data, k_item)}")
    
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
