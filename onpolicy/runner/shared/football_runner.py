import time
import numpy as np
import torch
from onpolicy.runner.shared.base_runner import Runner
from tqdm import tqdm

def _t2n(x):
    return x.detach().cpu().numpy()

class FootballRunner(Runner):
    def __init__(self, config):
        super().__init__(config)
        self.eval_episodes = self.all_args.eval_episodes

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            self.trainer.policy.eps_decay(episode, episodes)
            games_count, win, loss = 0, 0, 0

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)
                actions_env = np.squeeze(actions)
                # Obser reward and next obs
                
                obs, rewards, dones, infos = self.envs.step(actions_env)

                games_count += dones.sum() // self.num_agents
                for i, info in enumerate(infos):
                    if info['score_reward'] > 0: win += 1
                    elif info['score_reward'] < 0: loss += 1
                        
                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save(total_num_steps)

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Scenario {}-{} Algo {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}, eps {}.\n"
                        .format(self.all_args.scenario,
                                self.num_agents,
                                self.algorithm_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start)),
                                self.policy.epsilon))

                train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                train_infos["win_rate"] = win / games_count
                print("average episode rewards is {}, win/loss/total is {}/{}/{}, win rate is {}"
                    .format(train_infos["average_episode_rewards"], win, loss, games_count, train_infos["win_rate"]))
                
                self.log(train_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)
        
    def warmup(self):
        # reset env
        obs = self.envs.reset()
        share_obs = obs

        print(obs.shape)
        print(share_obs.shape)

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_states, rnn_states_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                            np.concatenate(self.buffer.obs[step]),
                            np.concatenate(self.buffer.rnn_states[step]),
                            np.concatenate(self.buffer.rnn_states_critic[step]),
                            np.concatenate(self.buffer.masks[step]))
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))

        return values, actions, action_log_probs, rnn_states, rnn_states_critic
    
    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)
        
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        rewards = np.expand_dims(rewards, -1)
        share_obs = obs

        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards, masks)

    # @torch.no_grad()
    # def eval(self, total_num_steps):

    #     eval_episode_rewards = []
    #     eval_obs = self.eval_envs.reset()

    #     n_iters = self.eval_episodes // self.n_eval_rollout_threads
    #     games_count, win, loss = 0, 0, 0

    #     for i in range(n_iters):
    #         eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]), dtype=np.float32)
    #         eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

    #         for eval_step in range(401):
    #             self.trainer.prep_rollout()
    #             eval_action, eval_rnn_states = self.trainer.policy.act(np.concatenate(eval_obs),
    #                                                                    np.concatenate(eval_rnn_states),
    #                                                                    np.concatenate(eval_masks),
    #                                                                    deterministic=True)
    #             eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))

    #             eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))

    #             eval_actions_env = np.squeeze(eval_actions)

    #             # Observe reward and next obs
    #             eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
    #             games_count += eval_dones.sum() // self.num_agents
    #             for i, info in enumerate(eval_infos):
    #                 if info['score_reward'] > 0: win += 1
    #                 elif info['score_reward'] < 0: loss += 1
                
    #             eval_episode_rewards.append(eval_rewards)

    #             eval_rnn_states[eval_dones == True] = np.zeros(
    #                 ((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
    #             eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
    #             eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

    #     mean_episode_rewards = np.array(eval_episode_rewards).sum(axis=0).mean()    # sum over steps and average over threads

    #     eval_env_infos = {
    #         'eval_average_episode_rewards': mean_episode_rewards,
    #         'eval_win_rate': win/games_count,
    #     }

    #     print("eval win/loss/total: {}/{}/{}, win rate: {}"
    #         .format(win, loss, games_count, win/games_count))

    #     if not self.all_args.eval_only: self.log(eval_env_infos, total_num_steps)
    
    @torch.no_grad()
    def eval(self, total_num_steps):

        assert self.eval_episodes % self.n_eval_rollout_threads == 0
        n_iters = self.eval_episodes // self.n_eval_rollout_threads
        
        win, loss = 0, 0
        from tqdm import tqdm
        for i in tqdm(range(n_iters)):
            eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

            finished = None
            eval_episode_rewards = []
            eval_obs = self.eval_envs.reset()

            for eval_step in range(401):
                self.trainer.prep_rollout()
                eval_action, eval_rnn_states = self.trainer.policy.act(
                    np.concatenate(eval_obs),
                    np.concatenate(eval_rnn_states),
                    np.concatenate(eval_masks),
                    deterministic=True)

                eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))

                eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))

                eval_actions_env = np.squeeze(eval_actions)

                # Observe reward and next obs
                eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
                
                eval_rewards = eval_rewards.reshape([-1, self.num_agents])
                if finished is None:
                    eval_r = eval_rewards[:,:self.num_agents]
                    finished = eval_dones.copy()
                else:
                    eval_r = (eval_rewards * ~finished)[:,:self.num_agents]
                    finished = eval_dones.copy() | finished
                eval_episode_rewards.append(eval_r.mean(-1))

                eval_rnn_states[eval_dones == True] = np.zeros(
                    ((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
                eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

                if finished.all() == True:
                    break
        
            # (step, rollout, ) -> (rollout, )
            episode_score = np.array(eval_episode_rewards).sum(axis=0)
            win += np.sum(episode_score==1)
            loss += np.sum(episode_score==-1)
            assert episode_score.shape[0] == self.eval_episodes / n_iters, f"{episode_score.shape[0]}!={self.eval_episodes}"
        
        eval_env_infos = {
            'win_rate': win/self.eval_episodes
        }

        print(f"eval win/loss/total: {win}/{loss}/{self.eval_episodes}, win rate: {win/self.eval_episodes}")

        if not self.all_args.eval_only: self.log(eval_env_infos, total_num_steps)
    