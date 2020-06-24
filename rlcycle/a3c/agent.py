

class TrajectoryRolloutWorker:
    """Worker that stores policy and runs environment trajectory
    
    Attributes:
        rank (int): rank of process
        env (gym.Env): gym environment to train on
        action_selector (ActionSelector): action selector given policy and state
        policy (BaseModel): policy for action selection

    """
    def __init__(self, rank: int, experiment_info: DictConfig, policy_cfg: DictConfig):
        self.rank = rank
        self.env = build_env(experiment_info)
        self.action_selector = build_action_selector(experiment_info)
        self.policy = build_model(policy_cfg)

    def run_trajectory(self):
        """Finish one env episode and return trajectory experience"""
        trajectory = dict(states=[], actions=[], rewards=[], next_states=[], dones=[])
        done = False:
        state = self.env.reset()
        while not done:
            # Carry out environment step
            action = self.action_selector(self.policy, state)
            next_state, reward, done, _ = self.env.step(action)
            
            # Store to trajectory
            trajectory["states"].append(state)
            trajectory["action"].append(action)
            trajectory["rewards"].append(reward)
            trajectory["next_states"].append(next_state)
            trajectory["dones"].append(dones)

            state = next_state

        trajectory_np = np.array(list(trajectory.values()))
        return trajectory_np

    def synchronize_policy(self, new_state_dict):
        self.policy.load_state_dict(new_state_dict)



class A3CAgent(Agent):

    def __init__(self):
        pass
    
    def _initialize(self):
        pass

    def run_gradient_parallel(self):
        """Run gradient parallel A3C """
        while self.update_step < self.experiment_info.max_update_steps:
            # Worker:
            #   1. sync policy with that of learner
            #   2. all workers run one trajectory
            #   3. each worker computes loss and gradent with trajectory
            #   4. Return gradients
            
            # Learner:
            #   1. Collect and apply gradients
            #   2. Update network   

    def run_data_parallel(self):
        """Run data parellel A3C (as in distributed data collection only)"""
        workers = [
            ray.remote(num_cpus=1)(
                TrajectoryRolloutWorker
            ).remote(worker_id, self.experiment_info, self.learner.model_cfg.actor)
            for worker_id in range(self.num_workers)
        ]

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
            
            # Synchronize worker policies
            policy_state_dict = self.learner.actor.state_dict()
            for worker in workers:
                worker.synchronize_policy.remote(policy_state_dict)

    def run_synchronous(self):
        """Run synchronous single worker (as in A2C)"""
        worker = TrajectoryRolloutWorker(
            0, self.experiment_info, self.learner.model_cfg.actor
        )

        while self.update_step < self.experiment_info.max_update_steps:
            # Run and retrieve trajectory
            trajectory = worker.run_trajectory()

            # Run update step with trajectory
            trajectory_tensor = self._preprocess_experience(trajectory_tensor)
            self.learner.update_model(trajectory)
            
            # Synchronize learner policy
            worker.synchronize_policy(self.learner.actor.state_dict())

    def train(self):
        """Run main training loop"""
        pass