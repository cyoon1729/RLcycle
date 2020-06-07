from typing import Callable, Tuple

from rlcycle.common.abstract.action_selector import ActionSelector
from rlcycle.common.abstract.agent import Agent
from rlcycle.dqn_base.learner import DQNLearner


class DQNBaseAgent(Agent):
    def __init__(self, args: dict, hyper_params: dict, model_cfg: dict, log_cfg: dict):
        Agent.__init__(self, args, hyper_params, model_cfg, log_cfg)
        self.use_n_step = self.hyper_params["n_step"] > 1
        self.transition_queue = deque(maxlen=self.hyper_params["n_step"])

    def _initialize(self):
        self.env = build_env(args)
        self.learner = build_learner(args, hyper_params, model_cfg)
        self.replay_buffer = ReplayBuffer(self.args, self.hyper_params)
        if self.args["use_per"]:
            self.learner = PERLearner(self.learner)
            self.replay_buffer = PrioritizedReplayBuffer(
                self.replay_buffer, self.hyper_params
            )
        self.action_selector = DQNActionSelector(self.device)
        self.action_selector = EpsGreedy(
            self.action_selector, self.env.action_space, self.hyper_params
        )

    def step(
        self, state: np.ndarray, action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.float64, bool]:
        action = self.action_selector(self.learner.network, state)
        next_state, reward, done, _ = self.env.step(action)
        return state, action, reward, next_state, done

    def train(self):
        for episode_i in range(self.args["total_episodes"]):
            state = self.env.reset()
            episode_reward = 0
            while not done:
                if self.args["render"]:
                    self.env.render()

                state, action, reward, next_state, done = self.step(state, action)
                episode_reward = episode_reward + reward

                if self.use_n_step:
                    transition = [state, action, reward, next_state, done]
                    self.transition_queue.append(transition)
                    if len(self.transition_queue) > self.hyper_params["n_step"]:
                        n_step_transition = preprocess_n_step(self.transition_queue)
                        self.replay_buffer.add(*n_step_transition)
                else:
                    self.replay_buffer.add(state, action, reward, next_state, done)

                if len(self.replay_buffer) > self.hyper_params["update_starts_size"]:
                    experience = self.replay_buffer.sample()
                    info = self.learner.update_model(experience)

                    if self.args["use_per"]:
                        q_loss, indices, new_priorities = info
                        self.replay_buffer.update_priorities(indices, new_priorities)
                    else:
                        q_loss = info

                self.logger.log_train(
                    self.learner.update_step, episode_i, q_loss, episode_reward
                )

            if episode_i % self.args["test_interval"]:
                self.test(episode_i)
                self.learner.save_params()

    def test(self, episode_i):
        print("====TEST START====")
        self.action_selector.exploration = False
        episode_rewards = []
        for test_i in range(self.args["test_num"]):
            episode_reward = 0
            while not done:
                if self.args["render"]:
                    self.env.render()

                state, action, reward, next_state, done = self.step(state, action)
                episode_reward = episode_reward + reward

            print(f"test {test_i}: {episode_reward}")
            episode_rewards.append(episode_reward)

        self.logger.log_test(episode_i, np.mean(episode_rewards))
        print("====TEST END====")
