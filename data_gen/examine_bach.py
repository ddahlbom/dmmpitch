import pretty_midi
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

################################################################################
# Utility functions
################################################################################
def plot_piano_roll(pm, start_pitch, end_pitch, fs=100):
    librosa.display.specshow(pm.get_piano_roll(fs)[start_pitch:end_pitch],
                             hop_length=1, sr=fs, x_axis='time', y_axis='cqt_note',
                             fmin=pretty_midi.note_number_to_hz(start_pitch))


################################################################################
# Main Script
################################################################################
# Construct the PrettyMIDI object
fs = 16000
pm = pretty_midi.PrettyMIDI('15ChristlaginTode.mid')

# Choose and instrument
inst_num = 53-1   # 'Ahh choir'
for i in pm.instruments:
    i.program = inst_num

# Look at Piano Roll
# plt.figure(figsize=(12,4))
# plot_piano_roll(pm, 24, 84)
# plt.show()

# Sonify
audio_data = pm.fluidsynth(fs=16000, sf2_path='/usr/share/soundfonts/FluidR3_GM.sf2')
# audio_data = pm.fluidsynth(fs=16000)
wav.write("bach.wav", 16000, audio_data)
