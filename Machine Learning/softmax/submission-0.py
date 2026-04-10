import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        max_z = np.max(z) 
        exp_z = np.exp(z-max_z) 
        sum_exp_z = np.sum(exp_z) 
        result = exp_z / sum_exp_z 
        return np.round(result,4) 