########################################################################################
# FILE: ex4_answers_practical.py
# WRITER : Aviel Shtern
# LOGIN : aviel.shtern
# ID: 206260499
# EXERCISE : Introduction to Machine Learning: Exercise 4 - PAC & Ensemble Method 2021
########################################################################################


from ex4_tools import *
from adaboost import *
import matplotlib.pyplot as plt

NUM_SAMPLES = 5000
NUM_TEST = 200
NOISE_RATIO = [0.0, 0.01, 0.4]
# NOISE_RATIO = [0.0]
T = 500
RANGE_T = np.arange(1, T + 1)
T_Q14 = [5, 10, 50, 100, 200, 500]


def run_separate_the_inseparable():
    """ Runs all the practical questions 13 - 17 """
    for i in range(len(NOISE_RATIO)):  # Repeat 13,14,15,16 for noise in {0, 0.01, 0.4}
        x_train, y_train = generate_data(NUM_SAMPLES, NOISE_RATIO[i])
        x_test, y_test = generate_data(NUM_TEST, NOISE_RATIO[i])
        ada_boost = AdaBoost(DecisionStump, T)
        D = ada_boost.train(x_train, y_train)
        Q13(ada_boost, x_train, y_train, x_test, y_test, NOISE_RATIO[i])
        Q14(ada_boost, x_test, y_test)
        Q15(ada_boost, x_train, y_train, x_test, y_test, NOISE_RATIO[i])
        Q16(ada_boost, x_train, y_train, NOISE_RATIO[i], D)


def Q13(ada_boost, x_train, y_train, x_test, y_test, noise):
    """
    generate 5000 samples without noise (i.e. noise_ratio=0). Train an Adaboost classifier over this data. Use the
    DecisionStump weak learner mentioned above, and T = 500. Generate another 200 samples without noise ("test set")
    and plot the training error and test error, as a function of T . Plot the two curves on the same figure.
    """
    training_error = []
    test_error = []
    for i in range(1, T + 1):
        training_error.append(ada_boost.error(x_train, y_train, i))
        test_error.append(ada_boost.error(x_test, y_test, i))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(RANGE_T, training_error, color="red", label="training_error")
    ax.plot(RANGE_T, test_error, color="blue", label="test_error")
    ax.set_xlabel("T - num of classifiers")
    ax.set_ylabel("error")
    ax.set_title(f"Q13 (for noise = {noise}) Ada-boost error as function of T")
    ax.legend()
    fig.show()


def Q14(ada_boost, x_test, y_test):
    """
    Plot the decisions of the learned classifiers with T ∈ {5, 10, 50, 100, 200, 500} together with the test data
    """
    for i in range(len(T_Q14)):
        plt.subplot(2, 3, i + 1)
        decision_boundaries(ada_boost, x_test, y_test, T_Q14[i])
    plt.show()


def Q15(ada_boost, x_train, y_train, x_test, y_test, noise)
    """
    Out of the different values you used for T , find Tˆ , the one that minimizes the test error. 
    Plot the decision boundaries of this classifier together with the training data.
    """
    test_errors = []
    for i in RANGE_T:
        curr_error = ada_boost.error(x_test, y_test, i)
        test_errors.append(curr_error)
    min_T = np.argmin(test_errors) + 1
    decision_boundaries(ada_boost, x_train, y_train, min_T)
    plt.title(f"Q15 - for noise = {noise}, T minimizes the test error= {min_T}")
    plt.show()


def Q16(ada_boost, x_train, y_train, noise, D):
    """
    Take the weights of the samples in the last iteration of the training (D^t). Plot the training set with size
    proportional to its weight in D^t
    """
    plt.subplot(2, 1, 1)
    decision_boundaries(ada_boost, x_train, y_train, T, weights=D)
    plt.title(f"Q16 - with noise = {noise} (we cannot see points in the upper plot!!)")
    plt.subplot(2, 1, 2)
    decision_boundaries(ada_boost, x_train, y_train, T, weights=(D / np.max(D) * 10))
    plt.title(f"Q16 - with noise = {noise} (normalized D now we see!)")
    plt.show()


if __name__ == "__main__":
    run_separate_the_inseparable()
