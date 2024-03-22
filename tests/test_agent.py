from agent import Agent

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
    pass

def test_jeffrey_update():
    pass
    
