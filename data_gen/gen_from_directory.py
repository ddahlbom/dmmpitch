import pretty_midi
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
import sys
sys.path.insert(0, '/home/dahlbom/research/ccarfac')
import pycarfac

################################################################################
# Basic Parameters
################################################################################

## File locations and names (ultimately set in CLI)
midi_path = "/home/dahlbom/research/dmm_pitch/data_gen/jsb_chorales"
file_name = "jsb_data" + ".bin"

## Synthesis Parameters
fs_aud = 12000
fs_sym = 20        # relate to frame-length of SAI

## Periphery Parameters
x_lo = 0.1  # normalized location for Greenwood
x_hi = 0.9  # normalized location for Greenwood
num_channels = 72

## SAI Parameters 

## Labeled Data from SAI parameters



################################################################################
# Main Script
################################################################################

# Make a list of all midi files in given directory
midi_files = glob.glob(midi_path + "/*.mid")
