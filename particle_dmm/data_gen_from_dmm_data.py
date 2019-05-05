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


################################################################################
# Other utility Functions
################################################################################
greenwood = lambda x : 165.4*(10**(2.1*x)-1)

################################################################################
# Basic Parameters
################################################################################

## File locations and names (ultimately set in CLI)
midi_path = "/home/dahlbom/research/dmm_pitch/data_gen/jsb_chorales_ib"
file_name = "jsb_data_ib"

## Synthesis Parameters
instrument_num = 53-1           # 'Ahh Choir'
pitch_lo = 13+20    # A1 (55 Hz)
pitch_hi = 76+20    # C7 (2093 Hz)
num_pitches = pitch_hi - pitch_lo + 1
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
program = 52    # Choral 'Ah'
fs_midi = 2     # Piano roll sampling frequency


################################################################################
# Main Script
################################################################################
# Load data
data = poly.load_data(poly.JSB_CHORALES)
training_seq_lengths = data['train']['sequence_lengths']
training_data_sequences = data['train']['sequences']
test_seq_lengths = data['test']['sequence_lengths']
test_data_sequences = data['test']['sequences']
val_seq_lengths = data['valid']['sequence_lengths']
val_data_sequences = data['valid']['sequences']


## Generate training data
num_training = training_data_sequences.shape[0]
for k in range(num_training):
    # Generate MIDI data from piano roll
    seq_len = test_seq_lengths[k].item()
    piano_roll = test_data_sequences[k,:seq_len,:].data.numpy()
    piano_roll_full = np.zeros((128,seq_len))
    piano_roll_full[MIDI_lo:MIDI_hi+1,:] += piano_roll.transpose().astype(int)
    piano_roll_full *= 64
    pm = piano_roll_to_pretty_midi(piano_roll_full.astype(float),
            fs=fs_midi, program=program)

    # Generate Audio from MIDI


    




################################################################################
# Old version
################################################################################
# midi_files = glob.glob(midi_path + "/*.mid")
# num_files = len(midi_files)
# 
# # Generate data for each file
# y_vals = np.zeros((0,num_pitches))
# x_vals = np.zeros((0,num_channels,frame_size_n), dtype=np.float32)
# block_num = 1
# block_time = 0.0
# for k, mf in enumerate(midi_files[:num_files]):
#     print("\n-------------------- Starting file {} of {} --------------------".format(k+1,num_files))
# 
#     # Load the data
#     pm = pretty_midi.PrettyMIDI(mf)
# 
#     # Extract the piano roll (i.e. the ground truth) 
#     end_time = pm.get_end_time()
#     block_time += end_time
#     if end_time > 150.0:
#         print("File {} has length {}.  It's too long! Skipping.".format(k+1,
#                 end_time))
#         continue
#     c_times = np.arange(0, end_time, 1./fs_sym)
#     c_times += frame_size_t/2.
#     c_times = c_times[:-1]  # last one should be outside of range after offset
#     y = pm.get_piano_roll(times=c_times)[pitch_lo:pitch_hi+1]
#     y = np.swapaxes(y, 0, 1)
#     y = np.where(y != 0, 1, 0)  #turn into binary 1/0 (on/off)
# 
#     # Set the instruments to choir
#     for i in pm.instruments:
#         i.program = instrument_num
# 
#     # Synthesize the audio
#     audio = pm.fluidsynth(fs=fs_aud,
#                           sf2_path='/usr/share/soundfonts/FluidR3_GM.sf2')
#     
#     # wav.write("carfac_killer_{}.wav".format(k), fs_aud, audio)
# 
#     ## Comment out to next double pound
#     # audio = audio[2*fs_aud:4*fs_aud] # isolate first two seconds
#     # num_samps = len(audio)
#     # t = np.arange(num_samps)/fs_aud
#     ##
# 
#     # Generate a Neural Activity Pattern (nap) from audio
#     nap, channel_cfs = pyc.carfac_nap(audio,
#                                       float(fs_aud),
#                                       num_sections=num_channels,
#                                       x_lo=x_lo,
#                                       x_hi=x_hi,
#                                       b=0.5)
# 
#     '''
#     ## Generate SAI
#     sai, frames_t, delays_t = pyc.carfac_sai(nap, 
#                                              fs_aud,
#                                              trig_win_t=trig_win_t,
#                                              adv_t = adv_t,
#                                              num_trig_win = num_trig_win)
#     print("Length of answers: ", len(y))
#     print("Number of frames:  ", len(frames_t))
# 
#     # Plotting below just used for demonstration. Delete later
# 
#     ## Plot NAP
#     fig = plt.figure()
#     p = 0.01 # offset scaling factor
#     skip_step = 4
#     ytick_vals = np.arange(num_channels)*p
#     ytick_vals = ytick_vals[::skip_step]
#     ytick_labels = ["{:.0f}".format(f) for f in np.flip(channel_cfs)]
#     ytick_labels = ytick_labels[::skip_step]
#     plt.yticks(ytick_vals, ytick_labels)
#     for k in range(num_channels):
#         plt.fill_between(t, 0, nap[k,:num_samps]+p*(num_channels-k), facecolor='w',
#                 edgecolor='k', linewidth=0.6)
#     
#     ## Animate SAI
#     num_frames = sai.shape[0]
#     animate_SAI(sai, fs_aud, frames_t, colormap=cm.binary,
#             adv_time=50)
#     '''
#     if calc_ac:
#         # generate frames directly from autocorrelation of nap frames 
#         x = gen_nap_ac_frames(nap, fs_aud, c_times, frame_size_t)
#     else: 
#         # generate frames directly from nap (no SAI, no AC)
#         x = gen_nap_frames(nap, fs_aud, c_times, frame_size_t)
# 
# 
#     x = np.array(x, dtype=np.float32)
#     x_vals = np.concatenate([x_vals, x])
#     y_vals = np.concatenate([y_vals, y])
# 
#     ## plot samples
#     # fig = plt.figure()
#     # num_frames = 10
#     # ax = []
#     # for k in range(num_frames):
#     #     ax.append(fig.add_subplot(2,5,k+1))
#     #     ax[k].imshow(x[k+50], aspect="auto", cmap=cm.binary)
#     # plt.show()
#     ##
# 
#     if block_time >= 400.0:
#         print("\nWriting block {}...".format(block_num))
#         f_name = file_name + "_{:04d}".format(block_num) + ".bin"
#         with open(f_name, "wb") as f:
#             pickle.dump((y_vals, x_vals), f)
#         y_vals = np.zeros((0,num_pitches))
#         x_vals = np.zeros((0,num_channels,frame_size_n), dtype=np.float32)
#         print("Finished writing.\n")
#         block_num += 1
#         block_time = 0.0
#      
# # clean up and write data
# print("label shapes: ", y_vals.shape)
# print("data shapes:  ", x_vals.shape)
# if (k+1) % 10 != 0:
#     print("\nWriting final block ({}) of output ...".format(block_num))
#     f_name = file_name + "_{:04d}".format(block_num) + ".bin"
#     with open(f_name, "wb") as f:
#         pickle.dump((y_vals, x_vals), f)
#     print("Finished writing.\n")
# 
# 
# 
