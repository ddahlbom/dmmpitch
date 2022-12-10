# dmmpitch
Using a Deep Markov Model (DMM) and pitch model, tied together with a particle filter, to make 
polyphonic pitch estimations.

One can often treat a pitch model as an observation probability which, together with a music (note 
transition) model, can be used to make pitch estimates.  An example of doing this for monophonic
pitch estimation can be found [here](https://github.com/analogouscircuit/particlepitch). This project
explores the possibility of extending this framework to polyphonic pitch estimation.  The key difficulty
here is the fact that polyphonic pitch space is so enormous (i.e. modeling transitions presents the problem
of combinatorial explosion).  Kirshnan et al.'s Deep Markov Model, a hybrid deep learning/probabilistic model,
was recently proposed as one approach to modeling this space. This study, originally presented at the
2019 meeting of the Acoustical Society of American in Louisville, explores this possibility. Several
different pitch models (Klapuri's model, as well as a hand-rolled MLP) are used to produce pitch
observation probabilities.  The [DMM model](https://github.com/pyro-ppl/pyro/blob/dev/examples/dmm.py) was 
drawn directly from the [PyTorch example](https://pyro.ai/examples/dmm.html).  

The model output is in the form of a piano roll.

![sample output](/images/sample_output.png)


## References

Dahlbom, D. A. and Braasch, J. (2019). "[Multiple F0 pitch estimation for musical applications using dynamic
Bayesian networks and learned priors](https://asa.scitation.org/doi/10.1121/1.5101633)," Journal of the Acoustical Society of America 145. [Abstract
to invited talk]

Klapuri, A. P. (2005). "[A perceptually-motivated multiple-F0 estimation method](https://ieeexplore.ieee.org/abstract/document/1540227)," IEEE Workshop on Applications of Signal Processing to Audio and Acoustics.

Krishnan, R. G., Shalit, U., Sontag, D. (2016). "[Structured Inference Networks for Nonlinear State
Space Models](https://arxiv.org/pdf/1609.09869.pdf)."
