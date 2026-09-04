import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        new_z = []
        for j in z:
            j = j - max(z)
            new_z.append(j)
        sumi = 0
        for i in new_z:
            sumi += np.exp(i)
        
        arr2 = []
        for i in new_z:
            final_softmax = (np.exp(i)/sumi)
            arr2.append(round(final_softmax,4))
        
        return arr2
        pass
