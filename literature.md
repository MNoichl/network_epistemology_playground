# Literature
This file contains a ‘survey’ of the existing literature on Bala-Goyal style agent-based models of scientific interaction. It also includes a first attempt at an abstract for our work.

This literature was kicked off by two papers by Zollman:
- Zollman's first papers demonstrate that lower connectivity can be epistemically beneficial (Zollman, 2007, 2010)

The existing follow-up literature can roughly be divided into three strands:
1. A strand on robustness:
    - robustness under changes in parameter settings (Rosenstock et al., 2016), 
    - robustness under changes in modelling choices (Kummerfeld & Zollman, 2016; Frey & Šešelja, 2018, 2020; 
2. A strand on conformity: (Zollman, 2010; Mohseni & Williams, 2021; Weatherall & O’Connor, 2021; Fazelpour & Steel, 2022)
3. A strand on epistemically impure agents (including financial interests) (Holman & Bruner, 2015; Weatherall, O’Connor & Bruner, 2020)

Below, the literature is separated into high and low priority papers. 

# Paper 

## Abstract 
Simulation studies in network epistemology of science have focused on whether and when high connectivity, convergence speed and conformity are epistemically beneficial at the collective level. In contrast, in this talk, we aim to complement the existing literature by using agent-based models and simulations to study whether and how the network topology affects collective epistemic performance. The existing literature typically focuses on three simple network topologies while empirical networks generally are scale-free, i.e., have an extremely unequal degree distribution. Our study can be divided into a theoretical part and an empirically-informed part. The theoretical approach focuses on artificial scale-free networks. Our results show that the degree-inequality does not affect the reliability at the collective level. The empirically-informed approach aims to test these theoretical findings using real-world scientific networks regarding the perceptron and the peptic ulcer. Starting from such a real-world network, we generate several randomized variants and investigate the epistemic performance of these (counterfactual) networks. These empirical corroborations falsify the theoretical finding: degree-equality would be epistemically beneficial for real-world networks. Hence, we find that although hypothesis that degree-equality produces reliable groups is not robust across the board, it seems robust in an empirically sensitive way. 

## Introduction
Social epistemologists and philosophers of science are increasingly interested in social structures and their impact on epistemic practices. In recent years, philosophers have used agent-based models to study whether and when high connectivity, convergence speed and conformity produce epistemically reliable groups. Zollman (2007, 2010) pioneered this research field and argued that ‘in many cases a community made up of less informed individuals is more reliable at learning correct answers’ and ‘there is a robust tradeoff between speed and reliability that may be impossible to overcome’. Hence, high connectivity and speed of convergence may reduce epistemic reliability. 

These findings have been scrutinized. 
On the one hand, Rosenstock et al. (2016) demonstrate that these results are not robust under changes in parameter settings (Rosenstock et al., 2016). They argue that the results only obtain in difficult learning situations, where good information is hard to come by. 
On the other hand, Frey and Šešelja (2018, 2020) illustrate that these results are also not robust under changes in idealizing assumptions even in difficult learning situations (also see Kummerfeld & Zollman, 2016).
The consensus seems to be that the optimal network structure is highly context dependent (see also, Zollman, 2013).

The epistemic value of modelling and simulations heavily depends on the empirical adequacy of these models and simulations (Weisberg, 2013; Ylikoski & Aydinonat, 2014; Bokulich, 2017; Frey and Šešelja, 2018, Frigg & Hartmann, 2020).[^*] From this perspective, the existing literature faces four interrelated challenges: 

1. Virtually all simulation studies restrict the size of the community to 10 agents -- (Rosenstock et al., 2016) is a notable exception, but they restrict the size to 100 agents;
2. Virtually all simulation studies restrict their analysis to two or three simplistic network structures (i.e., the cycle, wheel and complete network) even though these simplistic structures are rarely empirically observed -- (Zollman, 2007, Sect. 3.2) and (Weatherall & O’Connor, 2021) are notable exceptions; 
3. Virtually all empirical networks are scale-free (i.e., have an extremely unequal degree distribution) (Albert and Barabási, 2002); and
4. There are few attempts to empirically calibrate theoretical findings on artificial networks towards empirical networks (ANY EXCEPTIONS?). 

In this paper, we aim to address these challenges by studying whether and when degree-inequality affects the epistemic reliability of groups. Let us call this the *degree-inequality effect*: the (hypothesized) proposition that lower levels of degree-inequality produce more reliable groups. This contribution complements existing work that focuses on robustness analyses (Rosenstock et al., 2016; Frey and Šešelja, 2018, 2020), the effect of conformity (Zollman, 2010; Mohseni & Williams, 2021; Weatherall & O’Connor, 2021; Fazelpour & Steel, 2022), and the impact of epistemically impure agents (including financial interests) (Holman & Bruner, 2015; Weatherall, O’Connor & Bruner, 2020). 

To the best of our knowledge, there is one precursor that studies the degree-inequality effect:[^1] Zollman (2007, Sect. 3.1) finds that the wheel does better than the complete network, which suggests that degree-inequality might be epistemically beneficial. However, since the cycle does better than the wheel, he concludes that low connectivity (not degree-inequality) produces the epistemic benefits. In any case, we interpret this as tangible evidence that degree-equality might be epistemically beneficial. Zollman (2007, Sect. 3.2) goes on to consider all possible networks of size 6 and studies which network statistics are strong predictors for epistemic reliability. He finds that degree-inequality is not correlated with reliability and, hence, disconfirms the degree-inequality effect.[^2] However, this finding is vulnerable to some issues mentioned above: the size of the communities is very small and there is no attempt to empirically calibrate the finding towards empirical networks. One of our central contributions is that we attempt to empirically calibrate these findings. 

We propose a two-pronged analysis that uses both artificial networks and empirical networks. First, we use artificial networks to study whether and when degree-inequality effect obtains. Second, we use empirical networks to study whether more degree equality would improve the group’s epistemic reliability. 

Our simulation study finds that the degree-inequality effect obtains in the empirical networks but not in the artificial networks. This means that relation between degree-inequality and reliability is context-dependent -- which aligns with the contemporary consensus that the optimal network structure is highly context dependent. Indeed, on the one hand, the artificial networks demonstrate that the degree-inequality effect is not robust across the board. On the other hand, the empirical networks strongly suggest that degree-inequality effect obtains under realistic circumstances. 

`Optional: we could include a philsci discussion. Below are some thoughts.`

What conclusions should we draw from these dissonant findings: is the degree-inequality effect real or not? To address this question, we draw on the literature on idealization and robustness. ...

Note: I found the distinction between robustness across the board and *empirically sensitive robustness* by (Fazelpour & Steel, 2022, ‘Diversity, trust, and conformity: a simulation study’ -- Section 6.3) helpful. However, their discussion is focused on the relation between simulation studies and experiments involving human subjects. Moreover, their discussion does not feel well integrated into the philsci literature (although I don't know whether much philsci literature exists on this distinction). 

## The model

Stylized example:
- Individuals update both on empirical data and on testimonial evidence.
    - Simple Bayesian updating: the agent updates on the evidence from her own and others’ experiments (but does not conditionalise on the fact that others’ performed a particular experiment).
- New method or theory might be better than the current, well understood, method or theory
    - Probability of success of old theory is fixed; uncertainty regarding probability of success of new theory. 
    - If they think the old theory is better, they don't work on it because they already have a good understanding. 


`On scale-free networks`

‘Many network theorists assume that real-world networks are scale-free (although the empirical validity of this assumption is not undisputed (Broido and Clauset 2019)). Roughly stated, this means that most people have little influence, and that influence is concentrated. (Technically, scale-free networks are such that the in-degree distribution follows a power law, at least asymptotically.) Preferential attachment models are one popular way to explain the mechanism behind real-world scale-free networks (Albert and Barabási 2002), this mechanism incorporates the “rich get richer” principle. Despite reasonable disagreement, I decided to follow the trend in network theory and focus my investigation on scale-free networks that were generated using preferential attachment.’ (Duijf, unpublished draft on majority voting)

## Footnotes
[^*]: Bibliographical information:

- Bokulich, Alisa. 2017. ‘Models and Explanation’. In Springer Handbook of Model-Based Science, edited by Lorenzo Magnani and Tommaso Bertolotti, 103–18. Springer Handbooks. Cham: Springer International Publishing. https://doi.org/10.1007/978-3-319-30526-4_4.
- Frey, Daniel, and Dunja Šešelja. 2018. ‘What Is the Epistemic Function of Highly Idealized Agent-Based Models of Scientific Inquiry?’ Philosophy of the Social Sciences, April. https://doi.org/10.1177/0048393118767085.
- Frigg, Roman, and Stephan Hartmann. 2020. ‘Models in Science’. In The Stanford Encyclopedia of Philosophy, edited by Edward N. Zalta, Spring 2020. Metaphysics Research Lab, Stanford University.
- Weisberg, Michael. 2013. Simulation and Similarity: Using Models to Understand the World. Oxford University Press.
- Ylikoski, Petri, and N. Emrah Aydinonat. 2014. ‘Understanding with Theoretical Models’. Journal of Economic Methodology 21 (1): 19–36. https://doi.org/10.1080/1350178X.2014.886470.

[^1] The study by Weatherall & O’Connor (2021) also covers small-worlds networks with 50 agents and find that these small-world networks yield higher levels of polarization (compared with the cycle, wheel and complete network). But, their study focuses on clique structures rather than degree inequality.  

[^2]: Zollman (2007, Sect. 3.2) reports that the network density and clustering coefficient are strong predictors of reliability. We plan to expand our analysis to include these (and other) network statistics in future work. 



# Required reading

## Zollman. 2007. The Communication Structure of Epistemic Communities

- Main claims / Results:
    - ‘The surprising result of this analysis is that in many cases a community made up of less informed individuals is more reliable at learning correct answers.’
    - ‘The model suggests that there is a robust tradeoff between speed and reliability that may be impossible to overcome.’
- Notes:
    - The comparison between the wheel and the cycle concerns equal versus unequal connectivity (see Section 3.1 on ‘royal family’). Since the wheel is less reliable than the cycle, it seems like equal connectivity is most reliable. 
    - Zollman considers all possible networks of size 6 and studies strong predictors for reliability. It turns out that *network density* and *clustering coefficient* are strong predictors of reliability.
    - Moreover, *in-degree variance* is not correlated with reliability.
        - ‘suggesting that it is not the centrality of the wheel, but its high connectivity that results in its decrease in reliability’ 
    - Robustness to strings of bad results:
        - “Both of these results suggest that unconnected networks are more robust to the occasional string of bad results than the connected network because those strings are contained in a small region rather than spread to everyone in the network.” 
            - I personally don’t understand the two results that Zollman discusses on page 582-583, but it might be interesting to think about whether and how one could measure robustness to strings of bad results for any given network. The thought is that this might be computationally easier than running tons of simulations (for large networks).
        

## Zollman. 2010. The Epistemic Benefit of Transient Diversity
https://doi.org/10.1007/s10670-009-9194-6 
- Main claims:
    - Peptic ulcer ‘case study’

## Zollman. 2013. ‘Network Epistemology: Communication in Epistemic Communities’
https://doi.org/10.1111/j.1747-9991.2012.00534.x.
- Overview of work in network epistemology.
- Claim: ‘This research has revealed that a wide number of different communication structures are best, but what is best in a given situation depends on particular details of the problem being confronted by the group.’
- Covers different types of models: 
    - costly information exchange, 
    - (linear) averaging (DeGroot; Lehrer & Wagner), 
    - bounded averaging (Hegselmann & Krause), 
    - specialized knowledge (Anderson, 2011), 
    - bandit problems (Bala & Goyal; Zollman).


## Frey and Šešelja. 2018. ‘What Is the Epistemic Function of Highly Idealized Agent-Based Models of Scientific Inquiry?’ 
https://doi.org/10.1177/0048393118767085.

- Main claims:
    - Epistemic function of recent ABMs of scientific interaction is only heuristic or exploratory as opposed to explanatory
    - The how-possibly explanatory value is not epistemically valuable in the case of these ABMs
        - How-actually vs how-possibly: ‘In other words, rather than showing how something actually happens, they show how and under which circumstances the given phenomenon is possible.’
    - The ABMs can become useful as evidential and explanatory roles. 
        - ‘In particular, we illustrate this point by analysing the historical case study on peptic ulcer disease’ (Section 5).
- In Zollman’s model is not how-possibly valuable:
    - It represents just a possibility
    - We are not able to make relevant (or interesting) counterfactual inferences.
- Two types of robustness
    - Changes in parameter settings
    - Changes in idealizing assumptions
- Note: small community size (4-11 agents)


## Frey and Šešelja. 2020. ‘Robustness and Idealizations in Agent-Based Models of Scientific Interaction’
https://doi.org/10.1093/bjps/axy039.
- Focus on the context of difficult inquiry to assess whether connectivity is epistemically suboptimal in difficult inquiries (as Zollman's work suggests). 
- Focus on robustness analysis under changes in the idealizing assumptions. 
- Philosophical relevance: 
    - According to Weisberg (2007): ‘a *minimalist* model contains only those factors that *make a difference* to the occurrence of the phenomenon in question.’ Hence, if the phenomenon is not robust under changes in idealizing modelling choices then, the model is not minimal. 
    - But, the model is still exploratory. Normative insights require empirical callibrations (Martini & Pinto, 2017).
- Adjustments to Zollman's model:
    - Restless bandits (probability of success of theories changes over time)
    - New interaction mechanism:
        - criticism is truth conducive
        - it occurs between proponents of rivalling theories
        - trigger: ‘the receiver of information is affected by criticism every time she corrects her belief about the rivalling theory in a positive direction’
    - Theory choice:
        - Rational inertia towards current theory
        - Threshold below which alternative theories are considered equally promising. 
- Results: 
    - ‘Our results suggest that whether and to what extent the degree of connectedness of a scientific community impacts its efficiency is a highly context-dependent matter since different conditions deem strikingly different results.’
    - Figure 10 gives a nice visualization of their results!

## Rosenstock, Bruner, and O’Connor. 2016. ‘In Epistemic Networks, Is Less Really More?’ 
https://doi.org/10.1086/690717.
- On parameter robustness: the result that decreased connectivity is epistemically beneficial is not robust under changes in parameter settings:
    - the number of researchers
    - the number of experiments (per round)
    - the relative success rates of the two theories.
- Philosophical relevance:
    - ‘When good information is harder to come by, this is exactly the situation in which, for these models, it is potentially useful to decrease the amount of information flowing between agents at each time step.’
    - How-potentially explanations (rather than how-possibly explanations)
        - ‘The models are taken as potentially telling us something about the real world, rather than showing that some phenomenon is possible in principle.’
- Result:
    - ‘Less connectivity can improve inquiry only in a small parameter range in which learning is especially difficult: 
        - situations in which there are relatively few researchers, 
        - relatively small batches of information collected at a time (i.e., frequent social learning), and 
        - small differences between the success of the correct and incorrect theories that the researchers are comparing.’
    - ‘When inquiry is easier, decreased connectivity simply slows learning and provides no particular benefits.’
        - In particular, when relative success rates >0.025.
        - Benefits quickly vanish as the group size increases (at about 50). 
- Note
    - Restricted to the wheel and complete network
    - Restricted to network size of 4 to 100

# Optional reading?

## Kummerfeld, and Zollman. 2016. ‘Conservatism and the Scientific State of Nature’. 
https://doi.org/10.1093/bjps/axv013.
- Adjustments to Zollman's model:
    - Scientists can intentionally keep pursuing the worse theory
    - $\epsilon$-greedy agents instead of Bayesian agents.
- Results:
    - 

## Fazelpour and Steel. 2022. ‘Diversity, Trust, and Conformity: A Simulation Study’.
https://doi.org/10.1017/psa.2021.25.
- One of the most recent publications using the two-bandit model. 
- On diversity and conformity

## Zollman. 2010. ‘Social Structure and the Effects of Conformity’. 
https://doi.org/10.1007/s11229-008-9393-8.
- On conformity

## Mohseni and Williams. 2021. ‘Truth and Conformity on Networks’
https://doi.org/10.1007/s10670-019-00167-6.


## Weatherall and O’Connor. 2021. ‘Conformity in Scientific Networks’.
https://doi.org/10.1007/s11229-019-02520-2.
- On conformity


## Holman, and Bruner. 2015. ‘The Problem of Intransigently Biased Agents’. 
- Focus on adding intransigently biased agents (i.e., epistemically impure agents due to, e.g., financial interests) to a community of truth-seekers. 
- Focus on solution involving endogenous network formation.
- Results
    - Adding epistemically impure agents: higher connectivity leads to higher efficiency.
    - Adding endogenous network formation drastically increases efficiency. 

## Weatherall, O’Connor, and Bruner. 2020. ‘How to Beat Science and Influence People: Policymakers and Propaganda in Epistemic Networks’
https://doi.org/10.1093/bjps/axy062.
- On financial incentives

