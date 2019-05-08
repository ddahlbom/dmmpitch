import pretty_midi
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
import pickle
import glob
import scipy.io.wavfile as wav 
import sys
sys.path.insert(0, '/home/dahlbom/research/ccarfac')
sys.path.insert(0, '/home/dahlbom/research/dmm_pitch/particle_dmm')
import pycarfac as pyc
import polyphonic_data_loader as poly

################################################################################
# Functions for MIDI wrangling
################################################################################
def piano_roll_to_pretty_midi(piano_roll, fs=100, program=0):
    '''Convert a Piano Roll array into a PrettyMidi object
     with a single instrument.
    Parameters
    ----------
    piano_roll : np.ndarray, shape=(128,frames), dtype=int (NOTE: NEEDED FLOAT!)
        Piano roll of one instrument
    fs : int
        Sampling frequency of the columns, i.e. each column is spaced apart
        by ``1./fs`` seconds.
    program : int
        The program number of the instrument.
    Returns
    -------
    midi_object : pretty_midi.PrettyMIDI
        A pretty_midi.PrettyMIDI class instance describing
        the piano roll.
    '''
    notes, frames = piano_roll.shape
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    # pad 1 column of zeros so we can acknowledge inital and ending events
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

    # use changes in velocities to find note on / note off events
    velocity_changes = np.nonzero(np.diff(piano_roll).T)

    # keep track on velocities and note on times
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
        # use time + 1 because of padding above
        velocity = piano_roll[note, time + 1]
        time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = velocity
        else:
            pm_note = pretty_midi.Note(
                velocity=prev_velocities[note],
                pitch=note,
                start=note_on_time[note],
                end=time)
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0
    pm.instruments.append(instrument)
    return pm

################################################################################
# Generating frames from NAP (i.e. not using SAI) 
################################################################################
class MyNormalize(mcolors.Normalize):
    def __call__(self, value, clip=None):
        f = lambda x,a: (2*x)**a*(2*x<1)/2. +(2-(2*(1-1*x))**a)*(2*x>=1)/2.
        return np.ma.masked_array(f(value,0.5))

# def gen_nap_frames(nap, fs_aud, fs_sym
def gen_nap_frames(nap, fs_aud, times, frame_size_t):
    nap_len_n = len(nap)
    frame_size_n = int(fs_aud*frame_size_t)
    c_indices = [int(t*fs_aud) for t in times]
    pad_lower = int(frame_size_n//2)
    pad_upper = frame_size_n - pad_lower
    # dims are: center time, channel, frame length 
    frames = np.zeros( (len(times), nap.shape[0], frame_size_n) )
    for k, idx in enumerate(c_indices):
        frames[k,:,:] = nap[:,(idx-pad_lower):(idx+pad_upper)]
    return frames

def gen_nap_ac_frames(nap, fs_aud, times, frame_size_t):
    nap_len_n = len(nap)
    frame_size_n = int(fs_aud*frame_size_t)
    c_indices = [int(t*fs_aud) for t in times]
    pad_lower = int(frame_size_n//2)
    pad_upper = frame_size_n - pad_lower
    # dims are: center time, channel, frame length 
    frames = np.zeros( (len(times), nap.shape[0], frame_size_n) )
    for k, idx in enumerate(c_indices):
        for c in range(nap.shape[0]):   # channel index
            frames[k,c,:] = np.correlate(nap[c,(idx-pad_lower):(idx+pad_upper)],
                                         nap[c,(idx-pad_lower):(idx+pad_upper)],
                                         mode='same')
    return frames


def gen_nap_sac_frames(nap, fs_aud, times, frame_size_t):
    # Like above, but returns a summary-AC -- i.e. all channels summed
    nap_len_n = len(nap)
    frame_size_n = int(fs_aud*frame_size_t)
    c_indices = [int(t*fs_aud) for t in times]
    pad_lower = int(frame_size_n//2)
    pad_upper = frame_size_n - pad_lower
    # dims are: center time, channel, frame length 
    frames = np.zeros( (len(times), nap.shape[0], frame_size_n) )
    for k, idx in enumerate(c_indices):
        for c in range(nap.shape[0]):   # channel index
            frames[k,c,:] = np.correlate(nap[c,(idx-pad_lower):(idx+pad_upper)],
                                         nap[c,(idx-pad_lower):(idx+pad_upper)],
                                         mode='same')
    frames = np.sum(frames, axis=1)
    return frames[:,int(frame_size_n//2):]


################################################################################
# Other utility Functions
################################################################################
greenwood = lambda x : 165.4*(10**(2.1*x)-1)

################################################################################
# Basic Parameters
################################################################################

## File locations and names (ultimately set in CLI)
# midi_path = "/home/dahlbom/research/dmm_pitch/data_gen/jsb_chorales_ib"
# file_name = "jsb_data_ib"

## Synthesis Parameters
fs_aud = 12000 
frame_size_t = 0.05             # corresponds, ideally, to an fs_sym = 20 Hz
# derived quantities
dt = 1./fs_aud
frame_size_n = int(frame_size_t/dt)
frame_size_t = frame_size_n*dt  # redefine to account for rounding
fs_sym = 1.0/(frame_size_t)     # relate to frame-length of SAI

## Periphery Parameters
x_lo = 0.05  # 45 Hz
x_hi = 0.75  # 6050 Hz 
num_channels = 72

## SAI Parameters 
trig_win_t = frame_size_t 
adv_t = trig_win_t
num_trig_win = 3

## Other Params
calc_ac = False
MIDI_lo = 21
MIDI_hi = 21 + 87
num_pitches = MIDI_hi - MIDI_lo + 1
program = 52    # Choral 'Ah'
fs_midi = 2     # Piano roll sampling frequency
win_size_n = 2048   # size of window for ac -- centered in frame


################################################################################
# Main Script
################################################################################
if __name__=="__main__":
    # Load data
    data = poly.load_data(poly.JSB_CHORALES)
    data_categories = ['train', 'test', 'valid']

    # Set up the different data sets (training, validation, testing) so they can be
    # iterated over
    data_seqs = {} 
    seq_lengths = {} 
    for category in data_categories:
        data_seqs[category] = data[category]['sequences']
        seq_lengths[category] = data[category]['sequence_lengths']

    ## Generate training data
    for category in data_categories:
        y_vals = np.zeros((0,num_pitches))
        x_vals = np.zeros((0,int(win_size_n//2)), dtype=np.float32)
        seq_length = seq_lengths[category]
        data_seq   = data_seqs[category]
        # seq_length = seq_length[:10]
        # data_seq   = data_seq[:10]
        num_sequences = data_seq.shape[0]
        for k in range(num_sequences):
            print("________________________________________________________________________________")
            print("Started Chorale {} of {} for category {}".format(k+1,
                                                                    num_sequences, 
                                                                    category))
            print("________________________________________________________________________________")

            # Generate MIDI data from piano roll
            seq_len = seq_length[k].item()
            piano_roll = data_seq[k,:seq_len,:].data.numpy()
            piano_roll_full = np.zeros((128,seq_len))
            piano_roll_full[MIDI_lo:MIDI_hi+1,:] += piano_roll.transpose().astype(int)
            piano_roll_full *= 64   # make nonzero
            pm = piano_roll_to_pretty_midi(piano_roll_full.astype(float),
                    fs=fs_midi, program=program)

            # Generate Audio from MIDI
            audio = pm.fluidsynth(fs=fs_aud,
                                  sf2_path='/usr/share/soundfonts/FluidR3_GM.sf2')
            
            # Generate Neural Activity Pattern (nap) from audio 
            print("Calculating CARFAC NAP...")
            nap, channel_cfs = pyc.carfac_nap(audio,
                                              float(fs_aud),
                                              num_sections=num_channels,
                                              x_lo=x_lo,
                                              x_hi=x_hi,
                                              b=0.5)
            print("Finished.")

            # Generate frames from synthensized audio
            len_sig_n = len(audio)
            len_frame_n = len_sig_n/seq_len
            num_frames = int(len_sig_n/len_frame_n)
            c_times_n = np.arange(0,num_frames)*len_frame_n+int(len_frame_n//2)
            c_times_t = c_times_n/fs_aud
            win_size_t = win_size_n/fs_aud
            print("Calculating frame data...")
            x = gen_nap_sac_frames(nap, fs_aud, c_times_t, win_size_t)
            print("Finished.")
            assert len(x) == seq_len
            
            x = np.array(x, dtype=np.float32)
            x_vals = np.concatenate([x_vals, x])
            y_vals = np.concatenate([y_vals, piano_roll])

        # Write resulting data
        print("Writing results to disk...")
        f_name = "poly_synth_data_{}".format(category) + ".bin"
        with open(f_name, "wb") as f:
            pickle.dump((y_vals, x_vals), f)
        print("Finished!")

