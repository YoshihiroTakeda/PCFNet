import sys
sys.path.append('.')
import numpy as np
import torch
from pcfnet.models import MDN, PCFNet
from pcfnet.data_processor import DataProcessor

if __name__ == "__main__": 
    DataProcessor = DataProcessor(use_columns=['ra_arcmin', 'dec_arcmin', 'comoving_z', 'g-i', 'HSC_i_cut', 'HSC_g_cut', 'HSC_r_cut'])
    ids = ["1918"]
    X, index, y = DataProcessor.data_selection(ids)

    try:
        mdn = MDN(7, 10, 3)
        X_mdn = torch.cat(mdn(torch.from_numpy(X.to_numpy()).float()),axis=1).detach().numpy()
    except Exception as e:
        print('Error in MDN: ', e)
        sys.exit(1)

    X = np.hstack([X[['ra_arcmin', 'dec_arcmin', 'g-i', 'HSC_i_cut']].to_numpy(), X_mdn])
    dataloader = DataProcessor.get_dataloader(X, index, y)

    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = PCFNet(
            dim=14,
            k=2,
            neighbor=5,
            connection="concat",
            hidden_num=[16,32,64,128,256],
        ).to(device)
    except Exception as e:
        print('Error in Initialization: ', e)
        sys.exit(1)
    try:
        for batch in dataloader:
            xc, m, t = batch
            model.zero_grad()
            xc = xc.transpose(2,1).squeeze(1).to(device, non_blocking=True)
            m = m.transpose(2,1).to(device, non_blocking=True)
            t = t.to(device)
            x, c  = xc[:,:3,:], xc[:,3:,:]
            y = model(x, c, m)
            break
    except Exception as e:
        print('Error in Execution: ', e)
        sys.exit(1)

    print('Models Test Passed')
        