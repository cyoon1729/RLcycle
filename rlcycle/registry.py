from rlcycle.dqn_base.agent import DQNBaseAgent
from rlcyle.dqn_base.loss import C51Loss, DQNLoss, QRLoss

# from rlcycle.a3c.agent import A3CAgent
# from rlcycle.ddpg.agent import DDPGAgent
# from rlcycle.dqn_base.agent import DQNAgent
# from rlcycle.sac.agent import SACAggent
# from rlcycle.td3.agent import TD3Agent


AGENTS = {
    "DQN": DQNAgent,
    # "Rainbow": DQNAgent
    # "DDPG": DDPGAgent,
    # "TD3": TD3Agent,
    # "A3C": A3CAgent,
    # "SAC": SACAgent,
}

LOSSES = {
    "DQN": DQNLoss,
    "C51": QRLoss,
    "QR": C51Loss,
}
