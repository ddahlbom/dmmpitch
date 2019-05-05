"""
An implementation of a Deep Markov Model in Pyro based on reference [1].
This is essentially the DKS variant outlined in the paper. The primary difference
between this implementation and theirs is that in our version any KL divergence terms
in the ELBO are estimated via sampling, while they make use of the analytic formulae.
We also illustrate the use of normalizing flows in the variational distribution (in which
case analytic formulae for the KL divergences are in any case unavailable).

Reference:

[1] Structured Inference Networks for Nonlinear State Space Models [arXiv:1609.09869]
    Rahul G. Krishnan, Uri Shalit, David Sontag
"""

import argparse
import time
from os.path import exists

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
# Model structure (DMM)
################################################################################

class Emitter(nn.Module):
    """
    Parameterizes the bernoulli observation likelihood `p(x_t | z_t)`
    """

    def __init__(self, input_dim, z_dim, emission_dim):
        super(Emitter, self).__init__()
        # initialize the three linear transformations used in the neural network
        self.lin_z_to_hidden = nn.Linear(z_dim, emission_dim)
        self.lin_hidden_to_hidden = nn.Linear(emission_dim, emission_dim)
        self.lin_hidden_to_input = nn.Linear(emission_dim, input_dim)
        # initialize the two non-linearities used in the neural network
        self.relu = nn.ReLU()

    def forward(self, z_t):
        """
        Given the latent z at a particular time step t we return the vector of
        probabilities `ps` that parameterizes the bernoulli distribution `p(x_t|z_t)`
        """
        h1 = self.relu(self.lin_z_to_hidden(z_t))
        h2 = self.relu(self.lin_hidden_to_hidden(h1))
        ps = torch.sigmoid(self.lin_hidden_to_input(h2))
        return ps


class GatedTransition(nn.Module):
    """
    Parameterizes the gaussian latent transition probability `p(z_t | z_{t-1})`
    See section 5 in the reference for comparison.
    """

    def __init__(self, z_dim, transition_dim):
        super(GatedTransition, self).__init__()
        # initialize the six linear transformations used in the neural network
        self.lin_gate_z_to_hidden = nn.Linear(z_dim, transition_dim)
        self.lin_gate_hidden_to_z = nn.Linear(transition_dim, z_dim)
        self.lin_proposed_mean_z_to_hidden = nn.Linear(z_dim, transition_dim)
        self.lin_proposed_mean_hidden_to_z = nn.Linear(transition_dim, z_dim)
        self.lin_sig = nn.Linear(z_dim, z_dim)
        self.lin_z_to_loc = nn.Linear(z_dim, z_dim)
        # modify the default initialization of lin_z_to_loc
        # so that it's starts out as the identity function
        self.lin_z_to_loc.weight.data = torch.eye(z_dim)
        self.lin_z_to_loc.bias.data = torch.zeros(z_dim)
        # initialize the three non-linearities used in the neural network
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, z_t_1):
        """
        Given the latent `z_{t-1}` corresponding to the time step t-1
        we return the mean and scale vectors that parameterize the
        (diagonal) gaussian distribution `p(z_t | z_{t-1})`
        """
        # compute the gating function
        _gate = self.relu(self.lin_gate_z_to_hidden(z_t_1))
        gate = torch.sigmoid(self.lin_gate_hidden_to_z(_gate))
        # compute the 'proposed mean'
        _proposed_mean = self.relu(self.lin_proposed_mean_z_to_hidden(z_t_1))
        proposed_mean = self.lin_proposed_mean_hidden_to_z(_proposed_mean)
        # assemble the actual mean used to sample z_t, which mixes a linear transformation
        # of z_{t-1} with the proposed mean modulated by the gating function
        loc = (1 - gate) * self.lin_z_to_loc(z_t_1) + gate * proposed_mean
        # compute the scale used to sample z_t, using the proposed mean from
        # above as input the softplus ensures that scale is positive
        scale = self.softplus(self.lin_sig(self.relu(proposed_mean)))
        # return loc, scale which can be fed into Normal
        return loc, scale


class Combiner(nn.Module):
    """
    Parameterizes `q(z_t | z_{t-1}, x_{t:T})`, which is the basic building block
    of the guide (i.e. the variational distribution). The dependence on `x_{t:T}` is
    through the hidden state of the RNN (see the PyTorch module `rnn` below)
    """

    def __init__(self, z_dim, rnn_dim):
        super(Combiner, self).__init__()
        # initialize the three linear transformations used in the neural network
        self.lin_z_to_hidden = nn.Linear(z_dim, rnn_dim)
        self.lin_hidden_to_loc = nn.Linear(rnn_dim, z_dim)
        self.lin_hidden_to_scale = nn.Linear(rnn_dim, z_dim)
        # initialize the two non-linearities used in the neural network
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(self, z_t_1, h_rnn):
        """
        Given the latent z at at a particular time step t-1 as well as the hidden
        state of the RNN `h(x_{t:T})` we return the mean and scale vectors that
        parameterize the (diagonal) gaussian distribution `q(z_t | z_{t-1}, x_{t:T})`
        """
        # combine the rnn hidden state with a transformed version of z_t_1
        h_combined = 0.5 * (self.tanh(self.lin_z_to_hidden(z_t_1)) + h_rnn)
        # use the combined hidden state to compute the mean used to sample z_t
        loc = self.lin_hidden_to_loc(h_combined)
        # use the combined hidden state to compute the scale used to sample z_t
        scale = self.softplus(self.lin_hidden_to_scale(h_combined))
        # return loc, scale which can be fed into Normal
        return loc, scale


class DMM(nn.Module):
    """
    This PyTorch Module encapsulates the model as well as the
    variational distribution (the guide) for the Deep Markov Model
    """

    def __init__(self, input_dim=88, z_dim=100, emission_dim=100,
                 transition_dim=200, rnn_dim=600, num_layers=1, rnn_dropout_rate=0.0,
                 num_iafs=0, iaf_dim=50, use_cuda=True):
        super(DMM, self).__init__()
        # instantiate PyTorch modules used in the model and guide below
        self.emitter = Emitter(input_dim, z_dim, emission_dim)
        self.trans = GatedTransition(z_dim, transition_dim)
        self.combiner = Combiner(z_dim, rnn_dim)
        # dropout just takes effect on inner layers of rnn
        rnn_dropout_rate = 0. if num_layers == 1 else rnn_dropout_rate
        self.rnn = nn.RNN(input_size=input_dim, hidden_size=rnn_dim, nonlinearity='relu',
                          batch_first=True, bidirectional=False, num_layers=num_layers,
                          dropout=rnn_dropout_rate)

        # if we're using normalizing flows, instantiate those too
        self.iafs = [InverseAutoregressiveFlow(AutoRegressiveNN(z_dim, [iaf_dim])) for _ in range(num_iafs)]
        self.iafs_modules = nn.ModuleList(self.iafs)

        # define a (trainable) parameters z_0 and z_q_0 that help define the probability
        # distributions p(z_1) and q(z_1)
        # (since for t = 1 there are no previous latents to condition on)
        self.z_0 = nn.Parameter(torch.zeros(z_dim))
        self.z_q_0 = nn.Parameter(torch.zeros(z_dim))
        # define a (trainable) parameter for the initial hidden state of the rnn
        self.h_0 = nn.Parameter(torch.zeros(1, 1, rnn_dim))

        self.use_cuda = use_cuda
        # if on gpu cuda-ize all PyTorch (sub)modules
        if use_cuda:
            self.cuda()

    def get_sample_with_logprobs(self, z_prev, num):
        # Essentially run the model forward, getting samples of z_n and x_n.
        # Needs to be "seeded" with an initial z0.

        # Get distribution parameters
        z_loc, z_scale = self.trans(z_prev)
        z_dist = dist.Normal(z_loc, z_scale)

        # Sample latent space and get log-probability of result
        z_t = pyro.sample("z_gen_%d" % num,
                          z_dist)
        lp_z_t = z_dist.log_prob(z_t)

        # Get Emission Parameters
        emission_probs_t = self.emitter(z_t)
        x_dist = dist.Bernoulli(emission_probs_t)

        # Sample emission and get log-probability of result
        x_t = pyro.sample("x_gen_%d" % num,
                          x_dist)
        lp_x_t = x_dist.log_prob(x_t)

        return z_t, lp_z_t, x_t, lp_x_t


    def get_sample(self, z_prev, num):
        # Essentially run the model forward, getting samples of z_n and x_n.
        # Needs to be "seeded" with an initial z0.

        # Get distribution parameters
        z_loc, z_scale = self.trans(z_prev)
        z_dist = dist.Normal(z_loc, z_scale)

        # Sample latent space
        z_t = pyro.sample("z_gen_%d" % num,
                          z_dist)

        # Get Emission Parameters
        emission_probs_t = self.emitter(z_t)
        x_dist = dist.Bernoulli(emission_probs_t)

        # Sample emission
        x_t = pyro.sample("x_gen_%d" % num,
                          x_dist)

        return z_t.detach(), x_t.detach()

    def get_z1_dist(self, sequence):
        # Run a sequence through the guide to get the parameters for the
        # distribution of the first hidden state.  This can then be used to
        # start a sequence of samples.  
        # 
        # Must sypply a torch tensor that has the time axis reversed.

        rnn_output, _ = self.rnn(sequence.unsqueeze(0))
        z_prev = self.z_q_0
        z_loc, z_scale = self.combiner(z_prev, rnn_output[0,0,:])
        return dist.Normal(z_loc, z_scale)


    # the model p(x_{1:T} | z_{1:T}) p(z_{1:T})
    def model(self, mini_batch, mini_batch_reversed, mini_batch_mask,
              mini_batch_seq_lengths, annealing_factor=1.0, training=True):

        # this is the number of time steps we need to process in the mini-batch
        T_max = mini_batch.size(1)

        # register all PyTorch (sub)modules with pyro
        # this needs to happen in both the model and guide
        pyro.module("dmm", self)

        # set z_prev = z_0 to setup the recursive conditioning in p(z_t | z_{t-1})
        z_prev = self.z_0.expand(mini_batch.size(0), self.z_0.size(0))

        # we enclose all the sample statements in the model in a plate.
        # this marks that each datapoint is conditionally independent of the others
        with pyro.plate("z_minibatch", len(mini_batch)):
            # sample the latents z and observed x's one time step at a time
            for t in range(1, T_max + 1):
                # the next chunk of code samples z_t ~ p(z_t | z_{t-1})
                # note that (both here and elsewhere) we use poutine.scale to take care
                # of KL annealing. we use the mask() method to deal with raggedness
                # in the observed data (i.e. different sequences in the mini-batch
                # have different lengths)

                # first compute the parameters of the diagonal gaussian distribution p(z_t | z_{t-1})
                z_loc, z_scale = self.trans(z_prev)

                # then sample z_t according to dist.Normal(z_loc, z_scale)
                # note that we use the reshape method so that the univariate Normal distribution
                # is treated as a multivariate Normal distribution with a diagonal covariance.
                with poutine.scale(scale=annealing_factor):
                    z_t = pyro.sample("z_%d" % t,
                                      dist.Normal(z_loc, z_scale)
                                          .mask(mini_batch_mask[:, t - 1:t])
                                          .to_event(1))

                # compute the probabilities that parameterize the bernoulli likelihood
                emission_probs_t = self.emitter(z_t)
                # the next statement instructs pyro to observe x_t according to the
                # bernoulli distribution p(x_t|z_t)
                if training:
                    pyro.sample("obs_x_%d" % t,
                                dist.Bernoulli(emission_probs_t)
                                    .mask(mini_batch_mask[:, t - 1:t])
                                    .to_event(1),
                                obs=mini_batch[:, t - 1, :])
                else:
                    pyro.sample("obs_x_%d" % t,
                                dist.Bernoulli(emission_probs_t)
                                    .mask(mini_batch_mask[:, t - 1:t])
                                    .to_event(1))
                # the latent sampled at this time step will be conditioned upon
                # in the next time step so keep track of it
                z_prev = z_t

    # the guide q(z_{1:T} | x_{1:T}) (i.e. the variational distribution)
    def guide(self, mini_batch, mini_batch_reversed, mini_batch_mask,
              mini_batch_seq_lengths, annealing_factor=1.0):

        # this is the number of time steps we need to process in the mini-batch
        T_max = mini_batch.size(1)
        # register all PyTorch (sub)modules with pyro
        pyro.module("dmm", self)

        # if on gpu we need the fully broadcast view of the rnn initial state
        # to be in contiguous gpu memory
        h_0_contig = self.h_0.expand(1, mini_batch.size(0), self.rnn.hidden_size).contiguous()
        # push the observed x's through the rnn;
        # rnn_output contains the hidden state at each time step
        rnn_output, _ = self.rnn(mini_batch_reversed, h_0_contig)
        # reverse the time-ordering in the hidden state and un-pack it
        rnn_output = poly.pad_and_reverse(rnn_output, mini_batch_seq_lengths)
        # set z_prev = z_q_0 to setup the recursive conditioning in q(z_t |...)
        z_prev = self.z_q_0.expand(mini_batch.size(0), self.z_q_0.size(0))

        # we enclose all the sample statements in the guide in a plate.
        # this marks that each datapoint is conditionally independent of the others.
        with pyro.plate("z_minibatch", len(mini_batch)):
            # sample the latents z one time step at a time
            for t in range(1, T_max + 1):
                # the next two lines assemble the distribution q(z_t | z_{t-1}, x_{t:T})
                z_loc, z_scale = self.combiner(z_prev, rnn_output[:, t - 1, :])

                # if we are using normalizing flows, we apply the sequence of transformations
                # parameterized by self.iafs to the base distribution defined in the previous line
                # to yield a transformed distribution that we use for q(z_t|...)
                if len(self.iafs) > 0:
                    z_dist = TransformedDistribution(dist.Normal(z_loc, z_scale), self.iafs)
                    assert z_dist.event_shape == (self.z_q_0.size(0),)
                    assert z_dist.batch_shape == (len(mini_batch),)
                else:
                    z_dist = dist.Normal(z_loc, z_scale)
                    assert z_dist.event_shape == ()
                    assert z_dist.batch_shape == (len(mini_batch), self.z_q_0.size(0))

                # sample z_t from the distribution z_dist
                with pyro.poutine.scale(scale=annealing_factor):
                    if len(self.iafs) > 0:
                        # in output of normalizing flow, all dimensions are correlated (event shape is not empty)
                        z_t = pyro.sample("z_%d" % t,
                                          z_dist.mask(mini_batch_mask[:, t - 1]))
                    else:
                        # when no normalizing flow used, ".to_event(1)" indicates latent dimensions are independent
                        z_t = pyro.sample("z_%d" % t,
                                          z_dist.mask(mini_batch_mask[:, t - 1:t])
                                          .to_event(1))
                # the latent sampled at this time step will be conditioned upon in the next time step
                # so keep track of it
                z_prev = z_t

################################################################################
# Setup, training, and evaluation
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

    # how often we do validation/test evaluation during training
    # val_test_frequency = 50
    # # the number of samples we use to do the evaluation
    # n_eval_samples = 1

    # # package repeated copies of val/test data for faster evaluation
    # # (i.e. set us up for vectorization)
    # def rep(x):
    #     rep_shape = torch.Size([x.size(0) * n_eval_samples]) + x.size()[1:]
    #     repeat_dims = [1] * len(x.size())
    #     repeat_dims[0] = n_eval_samples
    #     return x.repeat(repeat_dims).reshape(n_eval_samples, -1).transpose(1, 0).reshape(rep_shape)

    # # get the validation/test data ready for the dmm: pack into sequences, etc.
    # # val_seq_lengths = rep(val_seq_lengths)
    # # test_seq_lengths = rep(test_seq_lengths)
    # # val_batch, val_batch_reversed, val_batch_mask, val_seq_lengths = poly.get_mini_batch(
    # #     torch.arange(n_eval_samples * val_data_sequences.shape[0]), rep(val_data_sequences),
    # #     val_seq_lengths, cuda=args.cuda)
    # # test_batch, test_batch_reversed, test_batch_mask, test_seq_lengths = poly.get_mini_batch(
    # #     torch.arange(n_eval_samples * test_data_sequences.shape[0]), rep(test_data_sequences),
    # #     test_seq_lengths, cuda=args.cuda)

    # # instantiate the dmm
    dmm = DMM(rnn_dropout_rate=args.rnn_dropout_rate, num_iafs=args.num_iafs,
              iaf_dim=args.iaf_dim, use_cuda=args.cuda)

    # setup optimizer
    adam_params = {"lr": args.learning_rate, "betas": (args.beta1, args.beta2),
                   "clip_norm": args.clip_norm, "lrd": args.lr_decay,
                   "weight_decay": args.weight_decay}
    adam = ClippedAdam(adam_params)

    # setup inference algorithm
    elbo = JitTrace_ELBO() if args.jit else Trace_ELBO()
    svi = SVI(dmm.model, dmm.guide, adam, loss=elbo)

    # now we're going to define some functions we need to form the main training loop

    # saves the model and optimizer states to disk
    def save_checkpoint():
        log("saving model to %s..." % args.save_model)
        torch.save(dmm.state_dict(), args.save_model)
        log("saving optimizer states to %s..." % args.save_opt)
        adam.save(args.save_opt)
        log("done saving model and optimizer checkpoints to disk.")

    # loads the model and optimizer states from disk
    def load_checkpoint():
        assert exists(args.load_opt) and exists(args.load_model), \
            "--load-model and/or --load-opt misspecified"
        log("loading model from %s..." % args.load_model)
        dmm.load_state_dict(torch.load(args.load_model))
        log("loading optimizer states from %s..." % args.load_opt)
        adam.load(args.load_opt)
        log("done loading model and optimizer states.")

    # prepare a mini-batch and take a gradient step to minimize -elbo
    # def process_minibatch(epoch, which_mini_batch, shuffled_indices):
    #     if args.annealing_epochs > 0 and epoch < args.annealing_epochs:
    #         # compute the KL annealing factor approriate for the current mini-batch in the current epoch
    #         min_af = args.minimum_annealing_factor
    #         annealing_factor = min_af + (1.0 - min_af) * \
    #             (float(which_mini_batch + epoch * N_mini_batches + 1) /
    #              float(args.annealing_epochs * N_mini_batches))
    #     else:
    #         # by default the KL annealing factor is unity
    #         annealing_factor = 1.0

    #     # compute which sequences in the training set we should grab
    #     mini_batch_start = (which_mini_batch * args.mini_batch_size)
    #     mini_batch_end = np.min([(which_mini_batch + 1) * args.mini_batch_size, N_train_data])
    #     mini_batch_indices = shuffled_indices[mini_batch_start:mini_batch_end]
    #     # grab a fully prepped mini-batch using the helper function in the data loader
    #     mini_batch, mini_batch_reversed, mini_batch_mask, mini_batch_seq_lengths \
    #         = poly.get_mini_batch(mini_batch_indices, training_data_sequences,
    #                               training_seq_lengths, cuda=args.cuda)
    #     # do an actual gradient step
    #     loss = svi.step(mini_batch, mini_batch_reversed, mini_batch_mask,
    #                     mini_batch_seq_lengths, annealing_factor)
    #     # keep track of the training loss
    #     return loss

    # helper function for doing evaluation
    def do_evaluation():
        # put the RNN into evaluation mode (i.e. turn off drop-out if applicable)
        dmm.rnn.eval()

        # compute the validation and test loss n_samples many times
        val_nll = svi.evaluate_loss(val_batch, val_batch_reversed, val_batch_mask,
                                    val_seq_lengths) / torch.sum(val_seq_lengths)
        test_nll = svi.evaluate_loss(test_batch, test_batch_reversed, test_batch_mask,
                                     test_seq_lengths) / torch.sum(test_seq_lengths)

        # put the RNN back into training mode (i.e. turn on drop-out if applicable)
        dmm.rnn.train()
        return val_nll, test_nll

    # if checkpoint files provided, load model and optimizer states from disk before we start training
    if args.load_opt != '' and args.load_model != '':
        load_checkpoint()

    #######################################
    # LOAD TRAINED MODEL AND SAMPLE FROM IT 
    #######################################
    # Basic parameters
    fs_aud = 16000  # sampling rate for audio rendering
    fig = plt.figure()
    ax_gt = fig.add_subplot(1,3,1)
    ax_estimated = fig.add_subplot(1,3,2)
    ax_sampled = fig.add_subplot(1,3,3)
    MIDI_lo = 21
    MIDI_hi = 21 + 87
    num_notes = MIDI_hi - MIDI_lo + 1
    
    # Select a testing set and collect data and initial distribution
    idx = 1 
    seq_len = test_seq_lengths[idx].item()
    piano_roll_gt = test_data_sequences[idx,:seq_len,:].data.numpy()
    piano_roll_gt_rev = np.ascontiguousarray(np.flip(piano_roll_gt, axis=1))
    z1_dist = dmm.get_z1_dist(torch.tensor(piano_roll_gt_rev))

    # Plot ground truth and render as audio
    piano_roll_gt_full = np.zeros((128,seq_len))
    piano_roll_gt_full[MIDI_lo:MIDI_hi+1,:] += piano_roll_gt.transpose().astype(int)
    piano_roll_gt_full *= 64
    pm = piano_roll_to_pretty_midi(piano_roll_gt_full.astype(float), fs=2, program=52)
    audio = pm.fluidsynth(fs=16000,
                          sf2_path='/usr/share/soundfonts/FluidR3_GM.sf2')
    wav.write("test_ground_truth_out.wav", 16000, audio)
    ax_gt.imshow(np.flip(piano_roll_gt_full, axis=0))

    # Generate Observation Probability Distributions for each step
    win_size = int(len(audio)/piano_roll_gt.shape[0])
    sig_len = len(audio)
    num_hops = piano_roll_gt.shape[0] 
    obs_probs = torch.zeros((num_hops, 2, num_notes))
    for k in range(num_hops):
        start_idx = int(k*len(audio)/num_hops)
        obs_probs[k,1,:] = torch.from_numpy(
                                pp.midi_probs_from_signal(
                                    audio[start_idx:start_idx+win_size], fs_aud,
                                    MIDI_lo, MIDI_hi
                                    )
                                )
        obs_probs[k,0,:] = 1. - obs_probs[k,1,:]

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

    # Calculate initial weights
    for p in range(num_particles):
        prob = calc_obs_probs(obs_probs[0,:,:], x[0,p,:])
        w[0,p] *= prob

    # Normalize weights
    w[0,:] = normalize_weights(w[0,:])
    # print("Initial normalized weights: \n", w[0,:])

    # Start main loop
    for f in range(1, num_hops):
        print("Starting on frame {}... ".format(f), end="")

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
            # prob = calc_obs_probs(obs_probs[f,:,:], x[f,p,:])
            num_same = torch.sum(x[f,p,:].type(torch.long) == piano_roll_gt[f]).item() 
            prob = ((num_same-78)/10)**2
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
    audio = pm.fluidsynth(fs=16000,
                          sf2_path='/usr/share/soundfonts/FluidR3_GM.sf2')
    wav.write("test_estimated_out.wav", 16000, audio)
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
    audio = pm.fluidsynth(fs=16000,
                          sf2_path='/usr/share/soundfonts/FluidR3_GM.sf2')
    wav.write("test_sampled_out.wav", 16000, audio)
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
