from agent import Agent, BetaAgent


def test_basics():
    agent = Agent(1)
    assert agent.id == 1
    assert agent.credence <= 1
    assert agent.credence >= 0
    assert agent.n_success == 0
    assert agent.n_experiments == 0


def test_experiment():
    agent1 = Agent(1)
    agent1.credence = 0.4
    agent1.experiment(10, 0.05)
    assert agent1.n_success == 0
    assert agent1.n_experiments == 0


def test_bayes_update():
    agent1 = Agent(1)
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
    agent = BetaAgent(1)
    agent.alpha = 0
    agent.beta = 0
    agent.beta_update(n_experiments=4, n_success=1, uncertainty=0.1)
    assert agent.alpha == 1
    assert agent.beta == 3
    assert agent.credence == 0.25

    agent.beta_update(n_success=11, n_experiments=16, uncertainty=0.1)
    assert agent.alpha == 12
    assert agent.beta == 8
    assert agent.credence == 0.6

    agent.beta_update(n_success=32, n_experiments=64, uncertainty=0.4)
    assert agent.alpha == 44
    assert agent.beta == 40
    assert round(agent.credence, 2) == 0.52


def test_jeffrey_update():
    pass
