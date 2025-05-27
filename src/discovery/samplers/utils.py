import numpy as np

def x2p(x, param_names):
    """
    Converts a list of parameter values `x` to a dictionary representation.

    Args:
        x (list): A list of parameter values.

    Returns:
        dict: A dictionary representation of the parameter values, where the keys are the parameter names and the values are the corresponding values from `x`.
    """
    # does not handle vector parameters
    return {par: val for par, val in zip(param_names, x)}

def p2x(p):
    """
    Convert a dictionary of values to a NumPy array.

    Parameters:
        p (dict): A dictionary containing values.

    Returns:
        numpy.ndarray: A NumPy array containing the values from the dictionary.
    """
    return np.array(list(p.values()), 'd')