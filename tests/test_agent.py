from agent import Agent, Bandit, BetaAgent, UncertaintyProblem


def test_basics():
    uncertainty_problem = UncertaintyProblem()
    agent = Agent(1, uncertainty_problem)
    assert agent.id == 1
    assert agent.credence <= 1
    assert agent.credence >= 0
    assert agent.n_success == 0
    assert agent.n_experiments == 0


def test_experiment():
    uncertainty_problem = UncertaintyProblem()
    agent1 = Agent(1, uncertainty_problem)
    agent1.credence = 0.4
    agent1.experiment(10, 0.05)
    assert agent1.n_success == 0
    assert agent1.n_experiments == 0


def test_bayes_update():
    uncertainty_problem = UncertaintyProblem()
    agent1 = Agent(1, uncertainty_problem)
    agent1.credence = 0.6
    n_experiments = 10
    n_success = 8
    n_failures = n_experiments - n_success
    uncertainty = 0.05
    correct_result1 = 1 / (
        1
        + ((1 - agent1.credence) / agent1.credence)
        * ((0.5 - uncertainty) / (0.5 + uncertainty)) ** (n_success - n_failures)
    )
    correct_result2 = 1 / (
        1
        + (1 - agent1.credence)
        * (
            ((0.5 - uncertainty) / (0.5 + uncertainty))
            ** (2 * n_success - n_experiments)
        )
        / agent1.credence
    )

    agent1.bayes_update(
        n_success=n_success, n_experiments=n_experiments, uncertainty=uncertainty
    )
    assert correct_result1 == agent1.credence
    assert correct_result2 == agent1.credence


def test_beta_update():
    bandit = Bandit(p_theories=[0.6, 0.7])
    agent = BetaAgent(1, bandit)
    agent.beliefs = [[0, 0], [0, 0]]
    agent.beta_update([[1, 4], [0, 0]])
    assert agent.beliefs[0] == [1, 3]

    agent.beta_update([[11, 16], [0, 0]])
    assert agent.beliefs[0] == [12, 8]

    agent.beta_update([[32, 64], [1, 2]])
    assert agent.beliefs[0] == [44, 40]
    assert agent.beliefs[1] == [1, 1]


def test_experiment_beta():
    bandit = Bandit(p_theories=[0.6, 0.7])
    agent = BetaAgent(1, bandit)
    agent.experiment(n_experiments=100)
    assert (agent.experiment_result != 0).any()
    assert (agent.experiment_result == 100).any()

    agent.beliefs = [[3, 3], [0, 3]]
    agent.experiment(n_experiments=100)
    assert (agent.experiment_result[1] == [0, 0]).all()
    assert agent.experiment_result[0][1] == 100


def test_jeffrey_update():
    pass
