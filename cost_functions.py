
import numpy as np

def cross_entropy(AL, Y):
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
    print(cross_entropy(AL, Y))