import ray
from omegaconf import DictConfig

from rlcycle.a2c.worker import TrajectoryRolloutWorker
from rlcycle.a3c.worker import ComputesGradients
from rlcycle.build import build_learner
from rlcycle.common.abstract.agent import Agent


class A3CAgent(Agent):
    """Asynchronous Advantage Actor Critic (A3C) agent

    Attributes:
        learner (Learner): learner for A3C
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

    def step(self):
        """A2C agent doesn't use step() in its training, so no need to implement"""
        pass

    def train(self):
        """Run gradient parallel training (A3C)."""
        ray.init()
        workers = []
        for worker_id in range(self.experiment_info.num_workers):
            worker = TrajectoryRolloutWorker(
                worker_id, self.experiment_info, self.learner.model_cfg.actor
            )

            # Wrap worker with ComputesGradients wrapper
            worker = ray.remote(num_cpus=1)(ComputesGradients).remote(
                worker, self.hyper_params, self.learner.model_cfg
            )
            workers.append(worker)

        gradients = {}
        for worker in workers:
            gradients[worker.compute_grads_with_traj.remote()] = worker

        while self.update_step < self.experiment_info.max_update_steps:
            computed_grads_ids, _ = ray.wait(list(gradients))
            if computed_grads_ids:
                # Retrieve computed gradients
                computed_grads = ray.get(computed_grads_ids[0])
                critic_grads, actor_grads = computed_grads

                # Apply computed gradients and update models
                self.learner.critic_optimizer.zero_grad()
                for param, grad in zip(self.learner.critic.parameters(), critic_grads):
                    param.grad = grad
                self.learner.critic_optimizer.step()

                self.learner.actor_optimizer.zero_grad()
                for param, grad in zip(self.learner.actor.parameters(), actor_grads):
                    param.grad = grad
                self.learner.actor_optimizer.step()

                self.update_step = self.update_step + 1

                # Synchronize worker models with updated models and get it runnin again
                state_dicts = dict(
                    critic=self.learner.critic.state_dict(),
                    actor=self.learner.actor.state_dict(),
                )
                worker = gradients.pop(computed_grads_ids[0])
                worker.synchronize.remote(state_dicts)
                gradients[worker.compute_grads_with_traj.remote()] = worker

        ray.shut_down()
