# python tests/test_dqn.py experiment_info.log_wandb=True
# python tests/test_c51.py experiment_info.log_wandb=True
# python tests/test_rainbow.py experiment_info.log_wandb=True
python tests/test_rainbow.py experiment_info.env.name=BreakoutNoFrameskip-v4 experiment_info.log_wandb=True experiment_info.total_num_episodes=500000
python tests/test_c51.py experiment_info.env.name=BreakoutNoFrameskip-v4 experiment_info.log_wandb=True experiment_info.total_num_episodes=500000 hyper_params.max_exploration_frame=100000
python tests/test_dqn.py experiment_info.env.name=BreakoutNoFrameskip-v4 experiment_info.log_wandb=True experiment_info.total_num_episodes=500000
# python tests/test_rainbow.py experiment_info.env.name=AsterixNoFrameskip-v4 experiment_info.log_wandb=True
# python tests/test_c51.py experiment_info.env.name=AsterixNoFrameskipv4 experiment_info.log_wandb=True
# python tests/test_dqn.py experiment_info.env.name=AsterixNoFrameskip-v4 experiment_info.log_wandb=True
