import sys
sys.path.append('.')
from tqdm import tqdm
import numpy as np
from pcfnet.data_processor import DataProcessor

if __name__ == "__main__": 
    data_processor = DataProcessor(
        use_columns=['ra_arcmin', 'dec_arcmin', 'comoving_z', 'g-i', 'HSC_i_cut', 'HSC_g_cut', 'HSC_r_cut'],
        )
    try:
        data_processor.assign_ids([["1918"]], ["5160"])
    except Exception as e:
        print('Error in assigning ids: ', e)
        sys.exit(1)
    try:
        ids = ["1918"]
        X, index, y = data_processor.data_selection(ids)
    except Exception as e:
        print('Error in data selection: ', e)
        sys.exit(1)
    try:
        X = np.hstack([X.to_numpy(), np.random.random((X.shape[0], 9))])
        dataloader = data_processor.get_dataloader(X, index, y)
    except Exception as e:
        print('Error in initialization of dataloader: ', e)
        sys.exit(1)
    try:
        for i, batch in tqdm(enumerate(dataloader)):
            
            pass
            # print(batch[0][0])
    except Exception as e:
        print('Error in Execution: ', e)
        sys.exit(1)

    print('DataProcessor Test Passed')