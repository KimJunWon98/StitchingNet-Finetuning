# modules/seed.py

import random
import os
import numpy as np
import torch

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # # (필요 시) 아래 옵션을 켜면 CUDNN에서 결정론적 연산이 활성화됨
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
