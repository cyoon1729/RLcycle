import ray
from omegaconf import DictConfig
from rlcycle.a2c.worker import TrajectoryRolloutWorker
from rlcycle.build import build_learner
from rlcycle.common.abstract.agent import Agent


class A2CAgent(Agent):
    """Synchronous Advantage Actor Critic (A2C; data parallel) agent

    Attributes:
        learner (Learner): learner for A2C
        update_step (int): update step counter

    """

    def __init__(
        self,
        experiment_info: DictConfig,
        hyper_params: DictConfig,
        model_cfg: DictConfig,
    ):
        Agent.__init__(self, experiment_info, hyper_params, model_cfg)
        self.update_step = 0

        self._initialize()

    def _initialize(self):
        """Set env specific configs and build learner."""
        self.experiment_info.env.state_dim = self.env.observation_space.shape[0]
        if self.experiment_info.env.is_discrete:
            self.experiment_info.env.action_dim = self.env.action_space.n
        else:
            self.experiment_info.env.action_dim = self.env.action_space.shape[0]
            self.experiment_info.env.action_range = [
                self.env.action_space.low.tolist(),
                self.env.action_space.high.tolist(),
            ]

        self.learner = build_learner(
            self.experiment_info, self.hyper_params, self.model_cfg
        )

    def train(self):
        """Run data parellel training (A2C)."""
        ray.init()
        workers = []
        for worker_id in range(self.num_workers):
            worker = ray.remote(num_cpus=1)(TrajectoryRolloutWorker).remote(
                worker_id, self.experiment_info, self.learner.model_cfg.actor
            )
            workers.append(worker)

        while self.update_step < self.experiment_info.max_update_steps:
            # Run and retrieve trajectories
            trajectories = ray.get(
                [worker.run_trajectory.remote() for worker in workers]
            )

            # Run update step with multiple trajectories
            trajectories_tensor = [
                self._preprocess_experience.remote(traj) for traj in trajectories
            ]
            self.learner.update_model(trajectories_tensor)
            self.update_step = self.update_step + 1

            # Synchronize worker policies
            policy_state_dict = self.learner.actor.state_dict()
            for worker in workers:
                worker.synchronize_policy.remote(policy_state_dict)
