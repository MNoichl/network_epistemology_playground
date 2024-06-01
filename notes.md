* Beta-distributions
    * We implemented the behaviour of the agent by following the highest mean. Other options could be to take the mode or to sample from the beta-distributions
* assymmetry in experimental behaviour

* Bayes Agent and Beta Agent seem to be working. But they seem to have convergence at different regions of the parameter space (for example n_agents, n_steps, n_experiments, uncertainty, network). But in general the beta agent converges much faster, although I am not sure it is being very successful.
    - It might be good to think about this and somehow identify the regions of convergence in the parameter space.
    - Does network distribution affect these regions of convergence?

* Jeffrey agent is not really working, but I will do my best (Ignacio)
    - Part of the problem is that the jeffrey update collides with what the model is doing
    - in the model what we have is that each agent receives the cummulative number of successes and failures from their neighbors as input
    - this version of jeffrey update seems to require as input a neighbor. It is doable though, but I will let it sleep for a bit.
    - Well I've been putting a couple of hours and although I am making progress it might take more time.
    - Made progress, but the line:         p_new_worse_given_E = (self.credence * (1 - p_E_given_new_better) / (1 - p_E)) has a division by zero some times.
    - I managed to make the jeffrey work in the emulation of Weisberg here: https://colab.research.google.com/drive/19IedjgytgXciXRcrzEMlPpqm-ZYivwfz?usp=sharing, but it does not show good results.
