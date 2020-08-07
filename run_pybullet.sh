python tests/test_ddpg.py experiment_info.env.name=HalfCheetahPyBulletEnv-v0  models=fujimoto_ddpg experiment_info.experiment_name=FujimotoDDPG hyper_params=fujimoto 
python tests/test_sac.py experiment_info.env.name=HalfCheetahPyBulletEnv-v0 
python tests/test_ddpg.py experiment_info.env.name=HalfCheetahPyBulletEnv-v0 

