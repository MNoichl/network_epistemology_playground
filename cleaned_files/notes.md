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

* Zollman 2007, section 3.2
    - He considers (in small scale) the clustering coefficient.
    - He also considers the network density (proportion of actual vs potential nodes), which is analogous to the prob in barbasi-albert. But you can compute it for any networks.
    - "It appears that sparsely connected networks have a much higher “inertia.” This inertia takes two forms. First, an unconnected network experiences less widespread change in strategy on a given round than a highly connected network....Second, unconnected networks are less likely to occupy precarious positions than connected ones." ... "Both of these results suggest that unconnected networks are more robust to the occasional string of bad results than the connected network because those strings are contained in a small region rather than spread to everyone in the network. This allows the small networks to maintain some diversity in behaviors that can result in the better action ultimately winning out if more accurate information is forthcoming."
    - "An inspection of the five most reliable and five fastest networks suggests that the features of a network that make it fast and those that make it accurate are very different (see Figure 5)." "Ultimately, there is no right answer to the question of whether speed or reliability is more important– it will depend on the circumstance. Although a small decrease in reliability can mean a relatively large increase in speed, in some cases such sacrifices may not be worth making."
    - "There are three assumptions that underlie this model which might cause some concern. They are: 1. The learning in our model is governed by the observation of payoffs. 2. There is a uninformative action whose expected payoff is well known by all actors. 3. The informative action can take on one of very few expected payoffs and the possibilities are known by all actors." -- > "Laudan suggests that theory choice is a problem of maximizing expected return. We ought to choose the theory that provides the largest expected problem solving ability. Since we have often pursued a particular project for an extended time before being confronted with a serious contender, we will have a very good estimate of its expected utility. However, we will be less sure about the new contender, but we could not learn without giving it a try."

* Rosenstock et. al. 2016.
    - "We show that previous results from epistemic network models (Zollman, 2007, 2010; Kummerfeld and Zollman, 2015) showing the benefits of decreased connectivity in epistemic networks are not robust across changes in parameter values. Our findings motivate discussion about whether and how such models can inform real-world epistemic communities. As we argue, only robust results from epistemic network models should be used to generate advice for the real-world, and, in particular, decreasing connectivity is a robustly poor recommendation."
    - "As we will argue, our exploration sharpens the original result. We find that in these models, less connectivity improves inquiry only in a small parameter range in which learning is especially difficult: situations in which there are relatively few researchers, relatively small batches of information collected at a time, and small differences between the success of the correct and incorrect theories that the researchers are comparing. When inquiry is easier, decreased connectivity simply slows learning and provides no particular benefits."
    - "In particular, we will argue that epistemic network models cannot give specific prescriptive advice to epistemic communities as to which communication structures are best for inquiry. Because results in these models are not robust across parameter or structural choices, they will yield very different advice for epistemic communities of different sorts, and for different problems these communities tackle. It is impractical, and usually impossible, however, for epistemic communities to know what sort of situation they in. Furthermore, a coarse grained assessment of the community will not do as, for these highly idealized models, the model-world match is not a close one."
    - "As we will also argue, however, this does not imply that epistemic network models are useless. Robust phenomena across these models are much more promising vis-´a-vis informing real world communities. The observation by Zollman (2010) that transient cognitive diversity improves inquiry, for example, is robust across models and seems to capture an important aspect of success in real epistemic communities. Our results suggest two more such robust phenomena. First, when inquiry is easier, network structure matter less for successful inquiry. Second, for any community, there are better ways to improve exploratory success than to decrease connectivity."
    - The presentation of the model is clearer than Zollman's original.
    - "In what follows, we present the results replicating Zollman’s 2007 simulations with a wider parameter space. We find that parameters for which there is a notable benefit to decreased network connectivity occupy a relatively small niche of the total space. Weran Bala and Goyal style simulations varying pB, n (the number of trials an agent performs each round), network configurations, and network sizes."
    - "When Does Network Structure Matter in the Models? In Bala and Goyal style models of epistemic networks, less connectivity can improve the accuracy of learning, but this only happens for certain areas of parameter space. In particular, the effect occurs under circumstances where learning is more ‘difficult’ for the agents in that the community has more trouble distinguishing between two alternative theories."
    - "The learning situation can be more difficult in this way when the parameters have the following features: 1. The two actions that agents may take are more similar in terms of success rates (low pB). The population size is smaller; and 3. The amount of data collected each round is smaller (low n)."
    - "To illustrate this claim consider figure 7. For each set of parameters—n, network size, and pB—we average convergence times for the cycle and wheel network using our data from Section 3. This average gives us a proxy for how difficult it is for networks of agents to converge under each set of parameter values. We then compare this measure of speed to the strength of the Zollman effect. The data is divided into smaller networks (6, 10, and 20) and larger ones (50, 100) since there is a significant difference in the strength of the effect in these two sets of networks. Note that the y-axis is on a logarithmic scale to make the trend more visible. As is evident, there is a clear correlation between networks that take more time to converge and those where connectivity matters. This supports the claim above that network structure matters more when inquiry is trickier." --> "There is something unintuitive about this observation. When good information is harder to come by, this is exactly the situation in which, for these models, it is useful to decrease the amount of information flowing between agents at each time step. The way to think about this is to observe that sparsity in epistemic networks can provide the very benefit that Zollman outlined (helping groups of agents to avoid preemptively converging to incorrect beliefs about the world), but it will only provide it when agents are already in a more difficult situation for inquiry. When agents have enough good data, decreasing connectivity only slows learning without providing any benefit."
    - "What sorts of real world recommendations do network epistemology models license? Obviously it would be a mistake to go from Zollman’s original work to the conclusion that real epistemic communities should generally decrease connectivity between agents, or from Kummerfeld and Zollman (2015) to the conclusion that epistemic communities should increase connectivity.14 At very least, our results indicate that such measures should only be helpful for communities confronted by a more difficult situation for inquiry in the ways outlined above. Yet even this is too strong. Before delving into the details, it is helpful to first discuss a distinction from Weisberg and Reisman (2008). In the course of this paper we have demonstrated that certain results in Zollman’s models lack what these authors call parameter robustness, or the invariance of an effect across possible parameter values. They differentiate this from structural robustness, or the invariance of an effect across varying structural assumptions in a model.15 Does the Zollman effect show structural robustness? Grim (2009) finds something similar to the Zollman effect for structurally different epistemic networks (in particular, individuals aren’t presented with just two actions but instead are placed on an ’epistemic landscape’). As discussed, Kummerfeld and Zollman (2015) find that under some structural changes—agents who are not Bayesian learners, and a different payoff structure—the Zollman effect is replicated. Under other structural changes, however— the introduction of naturally exploratory agents into epistemic networks—the Zollman effect reverses. Holman and Bruner (2015) look at Bala and Goyal style networks where some agents are not motivated by epistemic concerns. These ‘biased’ agents simply attempt to sway the scientific community to their views. Again, in these models connected, rather than less connected, networks are more successful. To summarize, the Zollman effect has some structural robustness, but other similar epistemic network models with structural tweaks find the opposite effect—that more connected networks are more successful."
    - "Yet since the Zollman effect displays neither structural nor parameter robustness, all these questions, including the impossible ones, must be dealt with to determine whether epistemic network models recommend a sparser or denser network."

* This Zollman paper uses bandits: https://www.journals.uchicago.edu/doi/full/10.1093/bjps/axv013

* Discussion about stop condition.

* TODOS: The spiel will be: "Those networks are all very theoretical, real life epistemic networks are centralized: what are the effects bla bla" Because the robustness issue was already developed by Rosenstock.
    * 1. (Max) Get the real life citation networks for 'perceptron' and 'peptic ulcer' with the relevant date windows. They both share the feature that an alternative theory/paradigm was better but the standard one remained for longer than it should.
    * 2. (Ignacio) Incorporate speed into the simulations. (Done)
    * Incorporate density and clustering. It might be a good idea to change the density of empirical networks randomly. 
    * 3. (Ignacio) Run simulations with rewiring. (In Progress)
    * (Max) take a look at the simulation code, check for bugs.
    * (Ignacio) The tricky part is that beta and bayes have different zones of convergence/success. In particular, regarding the uncertainty parameter.
    * Check the edge direction is working well.
    * Incorporate network weights, if there is time.

* Speed (Ignacio)
    - Its working well but the regression I believe is defined for floats and we have integers here
    - "ValueError: y data is not in domain of logit link function. Expected domain: [0.0, 1.0], but found [2.0, 1000.0]" --> this is just for the speed (n_steps) variable

* Rewiring (Ignacio)
    - Need to incorporate the rewiring in the direction of randomness into the function (Done)

* Plotting
    - I, Ignacio, made some progress but might need Max for a wrap up.