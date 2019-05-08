import argparse
import time
from os.path import exists
import sys

import numpy as np
import torch
import torch.nn as nn

import polyphonic_data_loader as poly
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.distributions import InverseAutoregressiveFlow, TransformedDistribution
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO
from pyro.nn import AutoRegressiveNN
from pyro.optim import ClippedAdam
from util import get_logger

import matplotlib.pyplot as plt
import pretty_midi
import scipy.io.wavfile as wav

import partpitchutils as pp
sys.path.insert(0, '/home/dahlbom/research/ccarfac')
import pycarfac as pyc
import os
# from pathlib import Path
# wd = Path.cwd()
# project_directory = wd.parents[0]
project_directory = os.path.curdir + '/../'
sys.path.insert(0, project_directory + 'dnn_front_end/')
sys.path.insert(0, project_directory + 'data_gen/')
import dnn_front_end_poly as dnn
import data_gen_from_dmm_data as datagen
import dmm as dmm_model


################################################################################
# Functions for particle filtering
################################################################################
def normalize_weights(weights):
    tot = torch.sum(weights)
    return weights/tot


def discrete_sample(p_vals):
    '''
    Takes an arbitrary discrete distribution and returns a randomly selected
    state.  works by dividing up the unit interval according to the given
    probabilities, sampling from a uniform distribution, and choosing the
    state based on into which bucket the uniform sample falls.
    '''
    p_vals = p_vals.numpy()
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
    particles = particles.numpy()
    weights  = weights.numpy()
    vals = np.unique(particles, axis=0)
    probs = np.zeros(len(vals), dtype=float)
    for k, w in enumerate(weights):
        idx = np.where(vals==particles[k])
        probs[idx[0][0]] += w
    return torch.from_numpy(vals), torch.from_numpy(probs)


def calc_obs_probs(p_vals, observation):
    # p_vals -- bernoulli probabilities for each state. p_vals[0,k] is the
    # probability of being off, p_vals[1,k] = 1 - p_vals[0,k] is the
    # probability of being on. observation is a piano roll slice
    prob = torch.exp( torch.sum( torch.log(
        p_vals[observation.type(torch.long),torch.arange(len(observation))]
                    )))
    return prob


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
# Main Procedure (setup and particle filtering) 
################################################################################
def main(args):
    # setup logging
    log = get_logger(args.log)
    log(args)

    data = poly.load_data(poly.JSB_CHORALES)
    training_seq_lengths = data['train']['sequence_lengths']
    training_data_sequences = data['train']['sequences']
    test_seq_lengths = data['test']['sequence_lengths']
    test_data_sequences = data['test']['sequences']
    val_seq_lengths = data['valid']['sequence_lengths']
    val_data_sequences = data['valid']['sequences']
    N_train_data = len(training_seq_lengths)
    N_train_time_slices = float(torch.sum(training_seq_lengths))
    N_mini_batches = int(N_train_data / args.mini_batch_size +
                         int(N_train_data % args.mini_batch_size > 0))

    log("N_train_data: %d     avg. training seq. length: %.2f    N_mini_batches: %d" %
        (N_train_data, training_seq_lengths.float().mean(), N_mini_batches))

    ## instantiate the dmm
    dmm = dmm_model.DMM(rnn_dropout_rate=args.rnn_dropout_rate, num_iafs=args.num_iafs,
              iaf_dim=args.iaf_dim, use_cuda=args.cuda)
    dmm.eval()

    # setup optimizer
    adam_params = {"lr": args.learning_rate, "betas": (args.beta1, args.beta2),
                   "clip_norm": args.clip_norm, "lrd": args.lr_decay,
                   "weight_decay": args.weight_decay}
    adam = ClippedAdam(adam_params)

    # setup inference algorithm
    elbo = JitTrace_ELBO() if args.jit else Trace_ELBO()
    svi = SVI(dmm.model, dmm.guide, adam, loss=elbo)

    # loads the model and optimizer states from disk
    def load_checkpoint():
        assert exists(args.load_opt) and exists(args.load_model), \
            "--load-model and/or --load-opt misspecified"
        log("loading model from %s..." % args.load_model)
        dmm.load_state_dict(torch.load(args.load_model))
        log("loading optimizer states from %s..." % args.load_opt)
        adam.load(args.load_opt)
        log("done loading model and optimizer states.")


    if args.load_opt != '' and args.load_model != '':
        load_checkpoint()


    #######################################
    # LOAD TRAINED MODEL AND SAMPLE FROM IT 
    #######################################
    ## Basic parameters
    fs_aud = 12000  # sampling rate for audio rendering
    fig = plt.figure()
    ax_gt = fig.add_subplot(1,3,1)
    ax_estimated = fig.add_subplot(1,3,2)
    ax_sampled = fig.add_subplot(1,3,3)
    MIDI_lo = 21
    MIDI_hi = 21 + 87
    num_notes = MIDI_hi - MIDI_lo + 1 

    ## Load the front-end model
    net = dnn.Net(ac_length=1024)
    net.eval()
    save_prefix = "dnn_frontend_poly"
    save_path   = project_directory + "dnn_front_end/saved_models/"
    net.load_state_dict(torch.load(save_path + save_prefix + ".pt"))
    
    ## Select a testing set and collect data and initial distribution
    idx = 1 
    seq_len = test_seq_lengths[idx].item()
    piano_roll_gt = test_data_sequences[idx,:seq_len,:].data.numpy()
    piano_roll_gt_rev = np.ascontiguousarray(np.flip(piano_roll_gt, axis=1))
    z1_dist = dmm.get_z1_dist(torch.tensor(piano_roll_gt_rev))

    ## Plot ground truth and render as audio
    piano_roll_gt_full = np.zeros((128,seq_len))
    piano_roll_gt_full[MIDI_lo:MIDI_hi+1,:] += piano_roll_gt.transpose().astype(int)
    piano_roll_gt_full *= 64
    pm = piano_roll_to_pretty_midi(piano_roll_gt_full.astype(float), fs=2, program=52)
    audio = pm.fluidsynth(fs=fs_aud,
                          sf2_path='/usr/share/soundfonts/FluidR3_GM.sf2')
    wav.write("test_ground_truth_out.wav", fs_aud, audio)
    ax_gt.imshow(np.flip(piano_roll_gt_full, axis=0))

    ## Generate Neural Activity Patterns 
    # Periphery Parameters
    x_lo = 0.05  # 45 Hz
    x_hi = 0.75  # 6050 Hz 
    num_channels = 72
    print("Generating NAP...")
    nap, channel_cfs = pyc.carfac_nap(audio,
                                      float(fs_aud),
                                      num_sections=num_channels,
                                      x_lo=x_lo,
                                      x_hi=x_hi,
                                      b=0.5)
    print("Finished.")

    ## Generate auto-correlated frames
    win_size_n = 2048
    len_sig_n = len(audio)
    len_frame_n = len_sig_n/seq_len
    num_frames = int(len_sig_n/len_frame_n)
    c_times_n = np.arange(0,num_frames)*len_frame_n+int(len_frame_n//2)
    c_times_t = c_times_n/fs_aud
    win_size_t = win_size_n/fs_aud
    print("Calculating frame data...")
    sac_frames = datagen.gen_nap_sac_frames(nap, float(fs_aud), c_times_t, win_size_t)
    print("Finished.")
    assert len(sac_frames) == seq_len


    ## Plot some sample frames
    # fig = plt.figure()
    # idcs = np.random.randint(seq_len,size=10)
    # for k in range(10):
    #     ax = fig.add_subplot(2,5,k+1)
    #     ax.plot(sac_frames[k])
    # plt.show()





    ## Generate Observation Probability Distributions for each step
    def midi_probs_from_signal_dnn(ac_frame, fs, MIDI_lo, MIDI_hi, net, ac_size=1024):
        assert len(ac_frame) == ac_size
        obs_probs = torch.zeros((2, MIDI_hi-MIDI_lo+1))
        # take the signal-section from middle of frame
        outputs = net(torch.unsqueeze(torch.tensor(ac_frame, requires_grad=False),0).float())
        sig = nn.Sigmoid()
        outputs = sig(outputs)
        obs_probs[1,:] = outputs[0].detach()    # re-squeeze
        obs_probs[0,:] = 1.0 - obs_probs[1,:]
        return obs_probs
        
    ## obs_probs for  dnn front-end
    win_size = int(len(audio)/piano_roll_gt.shape[0])
    sig_len = len(audio)
    num_hops = piano_roll_gt.shape[0] 
    assert num_hops == sac_frames.shape[0]
    obs_probs = torch.zeros((num_hops, 2, num_notes), requires_grad=False)
    dnn_ests  = torch.zeros((num_hops, num_notes), requires_grad=False)
    for k in range(num_hops):
        obs_probs[k,:,:] = midi_probs_from_signal_dnn(sac_frames[k], 
                                               fs_aud,
                                               MIDI_lo, 
                                               MIDI_hi, 
                                               net, 
                                               ac_size=1024)
        dnn_ests[k] = torch.where(obs_probs[k,1,:] > 0.5, 
                               torch.ones_like(obs_probs[k,1,:]), 
                               torch.zeros_like(obs_probs[k,1,:]))

    ## Plot some sample front-end outputs
    # fig = plt.figure()
    # idcs = np.random.randint(seq_len, size=10)
    # for k in range(10):
    #     ax = fig.add_subplot(2,5,k+1)
    #     ax.plot(obs_probs[idcs[k],1,:].numpy())
    #     ax.plot(piano_roll_gt[idcs[k]])
    # plt.show()

    # TEST: calculate the probability of the first notes
    piano_roll_gt = torch.from_numpy(piano_roll_gt).type(torch.long)
    # prob = torch.sum(torch.log(
    #         obs_probs[0,piano_roll_gt[0,:],torch.arange(num_notes)]))
    # print("Probability of first state: ", np.exp(prob))


    # Particle Filtering Parameters
    # num_particles = 10000     # worked!
    num_particles = 1000
    z_dim = 100
    x_dim = 88
    z = torch.ones((num_particles, z_dim), requires_grad=False)
    x = torch.ones((num_hops, num_particles, x_dim), requires_grad=False)
    w = torch.ones((num_hops, num_particles), requires_grad=False)

    # Generate initial particles
    count = 0
    for p in range(num_particles):
        z_prev = pyro.sample("init_z", z1_dist) 
        z[p,:], x[0,p,:] = dmm.get_sample(z_prev, p)
        num_same = torch.sum(x[0,p,:].type(torch.long) == piano_roll_gt[0]).item() 
        if num_same == 88:
            count += 1
    print("Got {} correct samples in step {}".format(count, 0))
    count = 0

    ## Calculate initial weights
    for p in range(num_particles):
        prob = calc_obs_probs(obs_probs[0,:,:], x[0,p,:])
        w[0,p] *= prob
    ## Using DNN
    # for p in range(num_particles):
    #     prob = calc_obs_probs(obs_probs[1,:], x[0,:])
    #     w[0,p] *= prob
    ## Uniform
    # w[0,:] = 1.0/w.shape[1]
    # def get_probs_from_dnn()

    # for p in range(num_particles):
    #     outputs = net(x[0,p,:])
    #     note_est = torch.where(outputs > 0.5,
    #                            torch.ones_like(outputs),
    #                            torch.zeros_like(outputs))


    # Normalize weights
    w[0,:] = normalize_weights(w[0,:])
    # print("Initial normalized weights: \n", w[0,:])

    # Start main loop
    piano_roll_gt_f = torch.tensor(piano_roll_gt, dtype=torch.double) # warning
    for f in range(1, num_hops):
        # print("Starting on frame {}... ".format(f), end="")
        # print("GT      :\n", piano_roll_gt_f[f])
        # print("DNN Est :\n", dnn_ests[k])

        # Sample new particles
        z_vals, z_probs = particles_to_dist(z, w[f-1])
        for p in range(num_particles):
            idx = discrete_sample(z_probs)
            z[p,:], x[f,p,:] = dmm.get_sample(z_vals[idx], p+f*num_particles)
            num_same = torch.sum(x[f,p,:].type(torch.long) == piano_roll_gt[f]).item() 
            if num_same == 88:
                count += 1
        print("Got {} correct samples in step {}\t".format(count, f), end='')
        count = 0
         
        # Calculate Weights -- probably bring this into loop above
        for p in range(num_particles):
            ## 
            prob = calc_obs_probs(obs_probs[f,:,:], x[f,p,:])

            ## Primitive observation-free model (more notes correct, higher prop)
            # num_same = torch.sum(x[f,p,:].type(torch.long) == piano_roll_gt[f]).item() 
            # prob = ((num_same-78)/10)**2

            w[f,p] *= prob

        # Normalize
        w[f,:] = normalize_weights(w[f,:])

        print("Done!")

    # Now pull out the most probable path
    piano_roll_estimated = np.zeros((num_hops, 88))
    for f in range(num_hops):
        idx = np.argmax(w[f,:].numpy())
        piano_roll_estimated[f,:] = x[f,idx,:].numpy()

    # Get MIDI from piano from sampled roll
    piano_roll_estimated_full = np.zeros((128, seq_len), dtype=int)
    piano_roll_estimated_full[20:108,:] += piano_roll_estimated.transpose().astype(int)
    piano_roll_estimated_full *= 64
    pm = piano_roll_to_pretty_midi(piano_roll_estimated_full.astype(float), fs=2, program=52)
    audio = pm.fluidsynth(fs=fs_aud,
                          sf2_path='/usr/share/soundfonts/FluidR3_GM.sf2')
    wav.write("test_estimated_out.wav", fs_aud, audio)
    ax_estimated.imshow(np.flip(piano_roll_estimated_full, axis=0))

    # Sample a random sequence starting at the same initial latent state 
    x_vals = []
    z_vals = []
    piano_roll_sampled = np.zeros((seq_len, 88))

    for k in range(seq_len):
        z_new, x_new = dmm.get_sample(z_prev, k)
        x_vals.append(x_new)
        z_vals.append(z_new)
        piano_roll_sampled[k,:] = x_new.data.numpy()
        z_prev = z_new
    
    # Get MIDI from piano from sampled roll
    piano_roll_sampled_full = np.zeros((128, seq_len), dtype=int)
    piano_roll_sampled_full[20:108,:] += piano_roll_sampled.transpose().astype(int)
    piano_roll_sampled_full *= 64
    pm = piano_roll_to_pretty_midi(piano_roll_sampled_full.astype(float), fs=1, program=52)
    audio = pm.fluidsynth(fs=fs_aud,
                          sf2_path='/usr/share/soundfonts/FluidR3_GM.sf2')
    wav.write("test_sampled_out.wav", fs_aud, audio)
    ax_sampled.imshow(np.flip(piano_roll_sampled_full, axis=0))

    plt.show()


# parse command-line arguments and execute the main method
if __name__ == '__main__':
    assert pyro.__version__.startswith('0.3.1')

    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-epochs', type=int, default=5000)
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.0003)
    parser.add_argument('-b1', '--beta1', type=float, default=0.96)
    parser.add_argument('-b2', '--beta2', type=float, default=0.999)
    parser.add_argument('-cn', '--clip-norm', type=float, default=20.0)
    parser.add_argument('-lrd', '--lr-decay', type=float, default=0.99996)
    parser.add_argument('-wd', '--weight-decay', type=float, default=2.0)
    parser.add_argument('-mbs', '--mini-batch-size', type=int, default=20)
    parser.add_argument('-ae', '--annealing-epochs', type=int, default=1000)
    parser.add_argument('-maf', '--minimum-annealing-factor', type=float, default=0.1)
    parser.add_argument('-rdr', '--rnn-dropout-rate', type=float, default=0.1)
    parser.add_argument('-iafs', '--num-iafs', type=int, default=0)
    parser.add_argument('-id', '--iaf-dim', type=int, default=100)
    parser.add_argument('-cf', '--checkpoint-freq', type=int, default=0)
    parser.add_argument('-lopt', '--load-opt', type=str, default='')
    parser.add_argument('-lmod', '--load-model', type=str, default='')
    parser.add_argument('-sopt', '--save-opt', type=str, default='')
    parser.add_argument('-smod', '--save-model', type=str, default='')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--jit', action='store_true')
    parser.add_argument('-l', '--log', type=str, default='dmm.log')
    args = parser.parse_args()

    main(args)
