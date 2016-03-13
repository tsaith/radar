import numpy as np

def power2db(p):
    """
    Power to db.
    p: normalized power.
    """
    return 10*np.log10(p)
