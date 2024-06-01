* Beta-distributions
    * We implemented the behaviour of the agent by following the highest mean. Other options could be to take the mode or to sample from the beta-distributions
* assymmetry in experimental behaviour

* Something seems fishy because if you look at change in self.credence by state for each agent, they dont seem to change.
* I tried the bayes agent again and it seems to be working well in this version.

* Jeffrey agent is not really working, but I will do my best (Ignacio
    - Part of the problem is that the jeffrey update collides with what the model is doing
    - in the model what we have is that each agent receives the cummulative number of successes and failures from their neighbors as input
    - this version of jeffrey update seems to require as input a neighbor. It is doable though, but I will let it sleep for a bit.)
