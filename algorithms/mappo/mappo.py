"""Multi-Agent Proximal Policy Optimization (MAPPO).

Centralized critic with decentralized actors.
Based on: https://arxiv.org/abs/2103.01955
from algorithms.mappo.distributions import DiagGaussianDistribution


class ObservationEncoder(nn.Module):
    """Encodes individual agent observations."""

    def __init__(self, obs_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class Actor(nn.Module):
    """Decentralized actor: maps local obs → action distribution."""

    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.encoder = ObservationEncoder(obs_dim, hidden_dim)
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.dist = DiagGaussianDistribution(action_dim)

    def forward(self, obs):
        features = self.encoder(obs)
        mean = self.mean_head(features)
        return self.dist.proba_distribution(mean, self.log_std)

    def get_action(self, obs, deterministic=False):
        dist = self.forward(obs)
        if deterministic:
            action = dist.mean
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def evaluate_actions(self, obs, actions):
        dist = self.forward(obs)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, entropy


class Critic(nn.Module):
    """Centralized critic: maps joint state → value."""

    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state):
        return self.net(state)


class MAPPO(BaseAlgorithm):
    """Multi-Agent PPO with centralized training, decentralized execution."""

    def __init__(
        self,
        env,
        eval_config,
        n_agents=8,
        n_steps=128,
        n_epochs=10,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        lr=3e-4,
        ema_decay=0.0,
        model_dir="models",
        video_dir=None,
        obs_mode="lidar",
        **kwargs,
    ):
        super().__init__(env, eval_config, model_dir, video_dir, obs_mode=obs_mode)
        self.n_agents = n_agents
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.lr = lr
        self.ema_decay = ema_decay

        obs_dim = env.observation_space(env.possible_agents[0]).shape[0]
        act_dim = env.action_space(env.possible_agents[0]).shape[0]
        state_dim = obs_dim * n_agents

        self.actor = Actor(obs_dim, act_dim).to(device)
        self.critic = Critic(state_dim).to(device)
        self.ema_actor = None
        if self.ema_decay > 0:
            self.ema_actor = copy.deepcopy(self.actor).to(device)
            self.ema_actor.eval()
            for param in self.ema_actor.parameters():
                param.requires_grad_(False)

        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr
        )

        self.buffer = RolloutBuffer(
            buffer_size=n_steps,
            observation_space=env.observation_space(env.possible_agents[0]),
            action_space=env.action_space(env.possible_agents[0]),
            n_agents=n_agents,
            device=device,
            gae_lambda=gae_lambda,
            gamma=gamma,
            n_envs=n_agents,
        )

    def learn(self, total_timesteps: int, start_step: int = 0):
        writer = SummaryWriter(log_dir=f"logs/{os.path.basename(self.model_dir)}")
        num_updates = total_timesteps // (self.n_steps * self.n_agents)
        start_update = (start_step // (self.n_steps * self.n_agents)) + 1
        best_reward = -float("inf")

        print(f"Training MAPPO for {total_timesteps} timesteps ({num_updates} updates)")
        print(f"  Starting from step {start_step} (Update {start_update})")
        print(f"  n_agents={self.n_agents}, n_steps={self.n_steps}, obs_mode={self.obs_mode}")

        global_step = start_step

        for update in range(start_update, num_updates + 1):
            self.buffer.reset()
            episode_rewards = []

            obs_dict = self.env.reset()[0]
            obs_list = [obs_dict.get(a, np.zeros(self.buffer.obs_shape)) 
                        for a in self.env.possible_agents]
            dones = np.zeros(self.n_agents)

            for step in range(self.n_steps):
                obs_array = np.stack(obs_list)
                obs_tensor = torch.FloatTensor(obs_array).to(device)

                # Build centralized state
                state = obs_array.reshape(1, self.n_agents, -1)

                with torch.no_grad():
                    actions, log_probs = self.actor.get_action(obs_tensor)
                    state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(device)
                    values = self.critic(state_tensor).squeeze()
                    if values.dim() == 0:
                        values = values.unsqueeze(0).expand(self.n_agents)

                actions_np = actions.cpu().numpy()
                actions_clipped = np.clip(actions_np, -1, 1)

                # Build action dict
                action_dict = {}
                for i, agent_id in enumerate(self.env.possible_agents):
                    if agent_id in self.env.agents:
                        action_dict[agent_id] = actions_clipped[i]

                if len(action_dict) == 0:
                    obs_dict = self.env.reset()[0]
                    obs_list = [obs_dict.get(a, np.zeros(self.buffer.obs_shape))
                                for a in self.env.possible_agents]
                    dones = np.zeros(self.n_agents)
                    continue

                next_obs_dict, rewards_dict, terms, truncs, infos = self.env.step(action_dict)

                # Collect rewards
                rewards = np.array([rewards_dict.get(a, 0.0) for a in self.env.possible_agents])
                episode_rewards.append(rewards.mean())

                # Episode starts
                new_dones = np.array([
                    float(terms.get(a, False) or truncs.get(a, False))
                    for a in self.env.possible_agents
                ])

                self.buffer.add(
                    obs_array, actions_clipped, rewards, dones, values, log_probs, state
                )

                dones = new_dones
                obs_list = [next_obs_dict.get(a, np.zeros(self.buffer.obs_shape))
                            for a in self.env.possible_agents]
                global_step += self.n_agents

                # If all done, reset
                if all(new_dones):
                    obs_dict = self.env.reset()[0]
                    obs_list = [obs_dict.get(a, np.zeros(self.buffer.obs_shape))
                                for a in self.env.possible_agents]
                    dones = np.zeros(self.n_agents)

            # Compute returns
            with torch.no_grad():
                last_obs = np.stack(obs_list)
                last_state = last_obs.reshape(1, -1)
                last_values = self.critic(
                    torch.FloatTensor(last_state).to(device)
                ).squeeze()
                if last_values.dim() == 0:
                    last_values = last_values.unsqueeze(0).expand(self.n_agents)

            self.buffer.compute_returns_and_advantage(last_values, dones)

            # PPO update
            policy_losses, value_losses, entropy_losses = [], [], []
            for epoch in range(self.n_epochs):
                for rollout_data in self.buffer.get(self.batch_size):
                    new_log_probs, entropy = self.actor.evaluate_actions(
                        rollout_data.observations, rollout_data.actions
                    )

                    state_flat = rollout_data.states.reshape(len(rollout_data.states), -1)
                    new_values = self.critic(state_flat).squeeze()

                    advantages = rollout_data.advantages
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                    ratio = (new_log_probs - rollout_data.old_log_prob).exp()
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_loss = nn.functional.mse_loss(new_values, rollout_data.returns)
                    entropy_loss = -entropy.mean()

                    loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        list(self.actor.parameters()) + list(self.critic.parameters()),
                        self.max_grad_norm,
                    )
                    self.optimizer.step()
                    self._update_ema_actor()

                    policy_losses.append(policy_loss.item())
                    value_losses.append(value_loss.item())
                    entropy_losses.append(entropy_loss.item())

            # Logging
            mean_reward = np.mean(episode_rewards) if episode_rewards else 0
            writer.add_scalar("train/mean_reward", mean_reward, global_step)
            writer.add_scalar("train/policy_loss", np.mean(policy_losses), global_step)
            writer.add_scalar("train/value_loss", np.mean(value_losses), global_step)
            writer.add_scalar("train/entropy", -np.mean(entropy_losses), global_step)
            if self.ema_actor is not None:
                writer.add_scalar("train/ema_decay", self.ema_decay, global_step)

            if update % 10 == 0:
                print(f"  Update {update}/{num_updates} | Step {global_step} | "
                      f"Reward: {mean_reward:.3f} | PL: {np.mean(policy_losses):.4f} | "
                      f"VL: {np.mean(value_losses):.4f}")

            if mean_reward > best_reward:
                best_reward = mean_reward
                self.save_model(os.path.join(self.model_dir, "best_model", "model.pth"))

            if update % 50 == 0:
                self.save_model(
                    os.path.join(self.model_dir, "checkpoints", f"model_step_{global_step}.pth")
                )

        writer.close()
        self.save_model(os.path.join(self.model_dir, "final_model.pth"))
        print(f"Training complete. Best reward: {best_reward:.3f}")
        return self

    def predict(self, observation, deterministic=False):
        obs_tensor = torch.FloatTensor(observation).to(device)
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        with torch.no_grad():
            action, _ = self.actor.get_action(obs_tensor, deterministic=deterministic)
        return action.cpu().numpy()

    def _update_ema_actor(self):
        """Track a smoothed actor copy for more stable evaluation/checkpointing."""
        if self.ema_actor is None:
            return

        with torch.no_grad():
            for ema_param, param in zip(self.ema_actor.parameters(), self.actor.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1.0 - self.ema_decay)

            for ema_buffer, buffer in zip(self.ema_actor.buffers(), self.actor.buffers()):
                ema_buffer.copy_(buffer)

    def save_model(self, path=None):
        if path is None:
            path = os.path.join(self.model_dir, "model.pth")
        torch.save({
            "actor": self.actor.state_dict(),
            "ema_actor": self.ema_actor.state_dict() if self.ema_actor is not None else None,
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "obs_mode": self.obs_mode,
            "ema_decay": self.ema_decay,
        }, path)

    def load_checkpoint(self, path: str):
        """Load model and optimizer state from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        if self.ema_actor is not None and checkpoint.get("ema_actor") is not None:
            self.ema_actor.load_state_dict(checkpoint["ema_actor"])
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        
        # Return the step number from the filename if possible
        try:
            filename = os.path.basename(path)
            if "step_" in filename:
                return int(filename.split("_")[-1].split(".")[0])
        except Exception:
            pass
        return 0

    @classmethod
    def load_model(cls, model_dir: str):
        path = os.path.join(model_dir, "best_model", "model.pth")
        if not os.path.exists(path):
            path = os.path.join(model_dir, "final_model.pth")
        return torch.load(path, map_location=device)
