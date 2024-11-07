import sys
sys.path.append('.')
import numpy as np
from pcfnet.data_processor import Feeder

if __name__ == "__main__": 
    rng = np.random.default_rng(3)
    data = rng.random((100, 14))
    index = [[2 , np.array([0,2,4])] for i in range(10)]
    label = rng.integers(2,size=(10, 1))

    try:
        feeder = Feeder(data, index, label)
    except Exception as e:
        print('Error in Initialization: ', e)
        sys.exit(1)
    try:
        batch = feeder.__getitem__(1)
        print(batch)
    except Exception as e:
        print('Error in Execution: ', e)
        sys.exit(1)

    print('Feeder Test Passed')