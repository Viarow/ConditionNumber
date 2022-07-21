import numpy as np 
from numpy.linalg import cond 
from numpy.linalg import det
from CommPy.utils import *
from CommPy.modulation import QAMModem

load_path = './data/QAM4_16to16/channel_matrices.npy'
H_list = np.load(load_path, allow_pickle=True)
Tx = QAMModem(16)
constellation = Tx.constellation
P = np.square(np.abs(constellation[0]))
N = H_list.shape[-1]

def get_abs_det(idx):
    H = H_list[idx, :, :]
    print(cond(H))
    HHt = np.matmul(H, hermitian(H))
    HHt_plus = np.eye(N) + P * HHt
    print(np.abs(det(HHt_plus)))