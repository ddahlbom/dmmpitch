import numpy as np
import scipy.fftpack as fft
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/dahlbom/research/ccarfac')
import pycarfac as pyc


################################################################################
# functions 
################################################################################
H_lp = lambda fs, k, K : 1.0/(0.108*fs*k/K + 24.7)

def calc_U_from_nap(nap, fs):
    num_channels = nap.shape[0]
    win_n = nap.shape[1]
    # This can probably be vectorized to avoid for loop. Investigate later.
    # Also, can't we exchange sum and fourier transform...think this is dumb
    Z = np.zeros((num_channels, 2*win_n))
    for c in range(num_channels):
        padded_input = np.pad(nap[c],(0,len(nap[c])), 'constant',
                constant_values=0)
        Z[c] = np.abs(fft.fft(padded_input))
    return np.sum(Z[:,:win_n], axis=0)


def calc_U_from_sig(signal, fs, win_n):
    assert len(signal) >= win_n, "Signal too short for specified window size!"
    half_win_n = int(win_n//2)
    mid_n = int(len(signal)//2)
    sig = signal[mid_n - half_win_n: mid_n + half_win_n]

    # Periphery parameters
    x_lo = 0.05  # 45 Hz
    x_hi = 0.75  # 6050 Hz 
    num_channels = 72
    nap, channel_cfs = pyc.carfac_nap(sig,
                                      float(fs),
                                      num_sections=num_channels,
                                      x_lo=x_lo,
                                      x_hi=x_hi,
                                      b=0.5)
    # This can probably be vectorized to avoid for loop. Investigate later.
    Z = np.zeros((num_channels, 2*win_n))
    for c in range(num_channels):
        padded_input = np.pad(nap[c],(0,len(nap[c])), 'constant',
                constant_values=0)
        Z[c] = np.abs(fft.fft(padded_input))
    return np.sum(Z[:,:win_n], axis=0)

def salience(U, fs, f_eval, win_n, b=-0.02):
    f_max = fs/2
    df = f_max/win_n 
    f_vals = np.arange(win_n)*df
    saliences = np.zeros(len(f_eval))
    K = U.shape[0]
    for k, f in enumerate(f_eval):
        in_bounds = True
        j = 1
        while True:
            idx_lo = int((j*f-(df/2))/df)
            idx_hi = max([int((j*f+(df/2))/df), idx_lo])
            idcs = np.arange(idx_lo, idx_hi+1)
            if idx_hi < win_n:
                weights = H_lp(fs, idcs, K)
                saliences[k] += np.max(U[idcs]*weights)
            else:
                saliences[k] *= fs*f
                break
            j += 1

    # normalize
    saliences = (1 + b*np.log(fs*f_eval))*saliences

    #scale = np.sum(saliences)
    scale = np.max(saliences)*1.1

    # now balance and return
    return saliences/scale

################################################################################
# 
################################################################################
if __name__=="__main__":
    ## Signal parameters 
    fs = 12000
    f0s = [195.9977, 261.6256, 329.6276]    # G3, C4, E4
    num_h = 8   # number of harmonics
    dur = 0.1   # duration in seconds

    ## Generate signal
    dt = 1./fs
    # t = np.arange(0.0, dur, dt)
    num_samples = 1024
    t = np.arange(num_samples)*dt
    sig = np.zeros_like(t)
    for f0 in f0s:
        for h in range(num_h):
            sig += ((1.0/(h+1))**0.9)*np.sin(2.0*np.pi*f0*(h+1)*t 
                                             + np.pi*2*np.random.random())
    sig /= np.max(sig)*2
    wav.write("input_signal.wav", fs, sig)

    ## Process through periphery
    U = calc_U_from_sig(sig, fs, num_samples)

    # calculate saliences
    f_eval = np.array([27.5 * (2**(k/12)) for k in range(88)])
    b_vals = np.arange(0, -0.05, -0.005)
    print(b_vals)
    fig = plt.figure()
    for k, b in enumerate(b_vals):
        ax = fig.add_subplot(2,5,k+1)
        saliences = salience(U, fs, f_eval, num_samples, b=b)
        ax.plot(f_eval, saliences)

    # plt.plot(f_vals, U)
    # plt.figure()
    # plt.plot(f_eval, saliences)
    plt.show()
