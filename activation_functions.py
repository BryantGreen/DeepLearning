

""" This is a work in progress.  """


import numpy as np


def softmax(x):
    """
    x - A matrix of shape (n,m)

    Returns:
    s - A matrix of shape (n,m)
    """
    
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=1, keepdims=True)
    s = x_exp / x_sum
    # Could be shortened to:
    #return np.exp(x) / np.sum(np.exp(x))
    return s

def relu(x):
    pass


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def test_sigmoid():
    x = np.array([1, 2, 3])
    print(sigmoid(x))
    # [0.73105858 0.88079708 0.95257413]
print(test_sigmoid())


x = np.array([
    [9, 2, 5, 0, 0],
    [7, 5, 0, 0 ,0]])
print("softmax(x) = " + str(softmax(x)))


   
    