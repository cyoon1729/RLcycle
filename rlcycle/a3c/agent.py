import numpy as np
from omegaconf import DictConfig, OmegaConf
import ray

from rlcycle.a2c.worker import TrajectoryRolloutWorker
from rlcycle.a3c.worker import ComputesGradients
from rlcycle.build import build_action_selector, build_learner
from rlcycle.common.abstract.agent import Agent
from rlcycle.common.utils.logger import Logger


class A3CAgent(Agent):
    """Asynchronous Advantage Actor Critic (A3C) agent

    Attributes:
        learner (Learner): learner for A3C
        update_step (int): update step counter
        action_selector (ActionSelector): action selector for testing
        logger (Logger): WandB logger
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

        self.action_selector = build_action_selector(
            self.experiment_info, self.use_cuda
        )

        # Build logger
        if self.experiment_info.log_wandb:
            experiment_cfg = OmegaConf.create(
                dict(
                    experiment_info=self.experiment_info,
                    hyper_params=self.hyper_params,
                    model=self.learner.model_cfg,
                )
            )
            self.logger = Logger(experiment_cfg)

    def step(self, state: np.ndarray, action: np.ndarray):
        """Carry out one environment step"""
        # A3C only uses this for test
        next_state, reward, done, _ = self.env.step(action)
        return state, action, reward, next_state, done

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
                computed_grads, step_info = ray.get(computed_grads_ids[0])
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

                if self.experiment_info.log_wandb:
                    log_dict = dict()
                    if step_info["worker_rank"] == 0:
                        log_dict["Worker 0 score"] = step_info["score"]
                    if self.update_step > 0:
                        log_dict["critic_loss"] = step_info["critic_loss"]
                        log_dict["actor_loss"] = step_info["actor_loss"]
                    self.logger.write_log(log_dict)

            if self.update_step % self.experiment_info.test_interval == 0:
                policy_copy = self.learner.get_policy(self.use_cuda)
                average_test_score = self.test(
                    policy_copy,
                    self.action_selector,
                    self.update_step,
                    self.update_step,
                )
                if self.experiment_info.log_wandb:
                    self.logger.write_log(
                        log_dict=dict(average_test_score=average_test_score),
                    )

                self.learner.save_params()

        ray.shut_down()
