

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


def cost_function(AL, Y):
    """    
    Inputs:
    AL -- probability vector corresponding to your label predictions, 
          shape (1, number of examples)
    Y - targets

    Returns:
    cross-entropy cost
    """
    m = Y.shape[1]
    
    cost = -1/m * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))
    
    # Squeeze just eliminates extra brackets from the calculation output.
    # or corrects shape.
    cost = np.squeeze(cost)
    assert(cost.shape ==())
    
    return cost

def test_cost_function():
    # We find from this test that all values of the probability vector are
    # required to be between 0 amd 1.
    # Values of 0 or 1 will cause an error.
    # This is correct.
    AL = np.array([0.001, 0.99, 0.95])
    Y = np.array([[0, 1, 2]])
    print( 'Y is shape {0}'.format(Y.shape))
    print(cost_function(AL, Y))
    
    