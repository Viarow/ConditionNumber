import numpy as np 
from numpy.linalg import cond 
import time
from CommPy.utils import *
from CommPy.modulation import QAMModem
from CommPy.cancelling.initialization import ZF_initial
from CommPy.cancelling.IterativeSIC import iterative_SIC
import matplotlib.pyplot as plt
from tqdm import tqdm

""" Global Variables"""
N = 16                 # number of users
M = 16                # number of receiving antennas
block_size = 500
mod = 16               # modulation order, for BPSK is 2
bit_width = int(np.log2(mod))  # number of bits in one symbol
SNRdB_range = np.arange(0, 18, 2)
H_mean = 0            # channel matrix distribution
H_var = 1             # channel matrix distribution
load_path = './data/QAM16_16to16/channel_matrices.npy'
num_iter = 6          # number of iterative SIC iterations
lld_k_Initial = np.log(1/mod * np.ones(mod))   # initial distribution of constellation symbols


""" TODO: Main Change Here """
def single_time_transmit(x_symbols, signal_power, var_noise, H, constellation):
    # compute noise
    noise_real = 0. + np.sqrt(0.5 * var_noise)*np.random.randn(M)
    noise_imag = 0. + np.sqrt(0.5 * var_noise)*np.random.randn(M)

    # channel: Rayleigh Fading
    y_symbols = np.matmul(H, x_symbols) + (noise_real + 1j * noise_imag)

    """ detection """
    X_Initial, var_Initial = ZF_initial(H, y_symbols, constellation, lld_k_Initial)
    xhat_mat, prob = iterative_SIC(X_Initial, var_Initial, var_noise, y_symbols, H, constellation, num_iter)
    xhat_indices = xhat_mat[num_iter-1]
    
    return xhat_indices


def main():

    # create modem
    modem = QAMModem(mod)
    constellation = modem.constellation

    # compute signal power
    signal_power = np.abs(constellation[0])

    # record BER after decoding and CRC pass rate
    SER_uncoded_main = np.zeros([SNRdB_range.shape[0], ])

    H_list = np.load(load_path, allow_pickle=True)
    idx = 0
    H = H_list[idx, :, :]

    start_time = time.time()
    """MAIN LOOPS"""
    for dd in range(0, SNRdB_range.shape[0]):
        var_noise = signal_power * H_var * np.power(10, -0.1*SNRdB_range[dd])
        SER_uncoded_block = np.zeros([block_size, ])

        for bb in tqdm(range(0, block_size)):
            # data bits in one packet
            x_bits = np.random.randint(0, 2, (bit_width*N, ))
            # no encoding
            x_indices, x_symbols = modem.modulate(x_bits)
            # transmit, detect
            xhat_indices = single_time_transmit(x_symbols, signal_power, var_noise, H, constellation)
            # SER
            SER_uncoded_block[bb] = bit_err_rate(x_indices, xhat_indices)

        # iteration over a block ends here
        SER_uncoded_main[dd] = np.mean(SER_uncoded_block)

        print("--- dd=%d --- SNR = %.1f dB  ---" % (dd, SNRdB_range[dd]))
        print(" Iterative SIC Symbol Error Rate: ")
        print(SER_uncoded_main)
    
    # iteration over all SNR ends here
    running_time = time.time() - start_time
    print('Running finished in %s seconds.'% (running_time))


if __name__ == '__main__':
    main()