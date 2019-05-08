import numpy as np
import matplotlib.pyplot as plt

################################################################################
# Global variables
################################################################################
half_step  = 2.**(1./12)
MIDI_nums = np.arange(128)
MIDI_freqs = np.array([27.5*(half_step**k) for k in range(-21,128-21)])   # A0 to C8
# print([k for k in zip(MIDI_nums, MIDI_freqs)])
MIDI_tau   = 1./MIDI_freqs
even = lambda x : True if (x%2)==0 else False


################################################################################
# Probability Functions
################################################################################
def discrete_sample(p_vals):
    '''
    takes an arbitrary discrete distribution and returns a randomly selected
    state.  works by dividing up the unit interval according to the given
    probabilities, sampling from a uniform distribution, and choosing the
    state based on into which bucket the uniform sample falls.
    '''
    cum = np.cumsum(p_vals)
    num_vals = len(cum)
    if np.abs(cum[-1]-1.) > 0.01:
        print(cum)
        print("Can't sample: not a valid distribution!")
        return 0
    old = 0
    u = np.random.uniform()
    for k in range(num_vals-1):
        if old <= u and u < cum[k]:
            return k
        old = cum[k]
    return num_vals-1 


def particles_to_dist(particles, weights):
    '''
    Converts particles into a histrogram/
    '''
    assert len(particles) == len(weights)
    vals = np.unique(particles)
    probs = np.zeros_like(vals, dtype=float)
    for k, w in enumerate(weights):
        idx = np.where(vals==particles[k])
        probs[idx[0][0]] += w
    idx = discrete_sample(probs)
    return vals, probs 


def q_sample(particles, weights):
    assert len(particles) == len(weights)
    vals = np.unique(particles)
    probs = np.zeros_like(vals, dtype=float)
    for k, w in enumerate(weights):
        idx = np.where(vals==particles[k])
        probs[idx[0][0]] += w
    idx = discrete_sample(probs)
    return vals[idx]


################################################################################
# Pitch Functions
################################################################################
def quad_interp(xs, ys):
    interp = lambda x : ( 
                ys[0]*(x - xs[1])*(x - xs[2])/((xs[0]-xs[1])*(xs[0]-xs[2])) + 
                ys[1]*(x - xs[0])*(x - xs[2])/((xs[1]-xs[0])*(xs[1]-xs[2])) + 
                ys[2]*(x - xs[0])*(x - xs[1])/((xs[2]-xs[0])*(xs[2]-xs[1])) )
    return interp
            

def signal_to_ac(signal, fs, half=False, mode='same'):
    signal = np.clip(signal, a_min=0, a_max = 100)  # HWR
    ac = np.correlate(signal, signal, mode=mode)
    tau = np.arange(len(signal))/fs
    len_n = len(signal)
    if even(len_n):
        tau -= ((len(signal))/fs)/2.
    else:
        tau -= ((len(signal)-1)/fs)/2.
    if half:
        len_half = int(len_n/2)
        ac  = ac[len_half:]
        tau = tau[len_half:]
    return ac, tau


def midi_probs_from_signal_ac(signal, fs, MIDI_lo, MIDI_hi):
    ac, tau_vals = signal_to_ac(signal, fs, half=True)
    MIDI_delays = MIDI_tau[MIDI_lo:MIDI_hi+1]
    saliencies = np.zeros_like(MIDI_delays)
    for k, tau in enumerate(MIDI_delays):
        idx = int(round(tau*fs))
        interp = quad_interp(tau_vals[idx-1:idx+2], ac[idx-1:idx+2])
        saliencies[k] = interp(tau) 
    return 0.95*(saliencies/np.max(saliencies))


################################################################################
# Test (if not loaded as module)
################################################################################
if __name__=="__main__":

    # Test signal parameters
    fs = 16000
    dt = 1./fs
    len_t = 0.1
    f0 = 220.0
    num_h = 5
    f_vals = [f0*k for k in range(1,num_h+1)]
    phase  = [2*np.pi*np.random.random() for k in range(num_h)]

    # Generate test signal
    len_n = int(fs*len_t)+1
    test_sig = np.zeros(len_n)
    t = np.arange(len_n)/fs
    for k in range(num_h):
        test_sig += np.sin(2*np.pi*f_vals[k]*t + phase[k]) 

    # Calculate AC -- just for checking, done internally in
    # midi_probs_from_signal(...)
    ac, tau_vals = signal_to_ac(test_sig, fs, half=True)

    # Get pitch probabilities
    MIDI_lo = 21    # lowest MIDI note to check for
    MIDI_hi = 21+87    # highest MIDI note to check for
    salience = midi_probs_from_signal(test_sig, fs, MIDI_lo, MIDI_hi)

    # Plot results
    fig = plt.figure()
    ax1 = fig.add_subplot(1,3,1)
    ax2 = fig.add_subplot(1,3,2)
    ax3 = fig.add_subplot(1,3,3)
    ax1.plot(t,test_sig)
    ax2.plot(tau_vals, ac)
    ax3.plot(MIDI_freqs[MIDI_lo:MIDI_hi+1], salience)

    plt.show() 
