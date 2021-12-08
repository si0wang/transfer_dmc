import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import itertools

import utils
from encoder import make_encoder
from decoder import make_decoder
from dynamics_model import DynamicsModel, RewardModel, DynamicsActionEncoder, RewardActionEncoder

LOG_FREQ = 10000


def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
    """Apply squashing function.
    See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
    """
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class Actor(nn.Module):
    """MLP actor network."""
    def __init__(
        self, obs_shape, action_shape, hidden_dim, encoder_type,
        encoder_feature_dim, log_std_min, log_std_max, num_layers, num_filters
    ):
        super().__init__()

        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters
        )

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # self.trunk = nn.Sequential(
        #     nn.Linear(self.encoder.feature_dim, hidden_dim), nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        #     nn.Linear(hidden_dim, 2 * action_shape[0])
        # )

        self.actor_head = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )

        self.actor_nn = nn.Linear(hidden_dim, 2 * action_shape[0])

        self.outputs = dict()
        self.apply(weight_init)

    def forward(
        self, obs, compute_pi=True, compute_log_pi=True, detach_encoder=False
    ):
        obs = self.encoder(obs, detach=detach_encoder)
        # print(obs.shape)
        # mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        latent = self.actor_head(obs)
        mu, log_std = self.actor_nn(latent).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
            self.log_std_max - self.log_std_min
        ) * (log_std + 1)

        self.outputs['mu'] = mu
        self.outputs['std'] = log_std.exp()

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std

    def log(self, L, step, log_freq=LOG_FREQ):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_actor/%s_hist' % k, v, step)

        L.log_param('train_actor/fc1', self.actor_head[0], step)
        L.log_param('train_actor/fc2', self.actor_head[2], step)
        L.log_param('train_actor/fc3', self.actor_nn, step)


class Actor_Finetune(nn.Module):
    """MLP actor network."""
    def __init__(
        self, obs_shape, action_shape, hidden_dim, encoder_type,
        encoder_feature_dim, log_std_min, log_std_max, num_layers, num_filters
    ):
        super().__init__()

        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters
        )

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # self.trunk = nn.Sequential(
        #     nn.Linear(self.encoder.feature_dim, hidden_dim), nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        #     nn.Linear(hidden_dim, 2 * action_shape[0])
        # )

        self.actor_head = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )

        self.actor_nn = nn.Linear(hidden_dim, 2 * action_shape[0])

        self.outputs = dict()
        self.apply(weight_init)

    def forward(
        self, obs, compute_pi=True, compute_log_pi=True, detach_encoder=False
    ):
        obs = self.encoder(obs, detach=detach_encoder)
        # print(obs.shape)
        # mu, log_std = self.trunk(obs).chunk(2, dim=-1)
        with torch.no_grad():
            latent = self.actor_head(obs)
        mu, log_std = self.actor_nn(latent).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
            self.log_std_max - self.log_std_min
        ) * (log_std + 1)

        self.outputs['mu'] = mu
        self.outputs['std'] = log_std.exp()

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std

    def log(self, L, step, log_freq=LOG_FREQ):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_actor/%s_hist' % k, v, step)

        L.log_param('train_actor/fc1', self.actor_head[0], step)
        L.log_param('train_actor/fc2', self.actor_head[2], step)
        L.log_param('train_actor/fc3', self.actor_nn, step)


class QFunction(nn.Module):
    """MLP for q-function."""
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=1)
        return self.trunk(obs_action)


class Critic(nn.Module):
    """Critic network, employes two q-functions."""
    def __init__(
        self, obs_shape, action_shape, hidden_dim, encoder_type,
        encoder_feature_dim, num_layers, num_filters
    ):
        super().__init__()


        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters
        )

        self.Q1 = QFunction(
            self.encoder.feature_dim, action_shape[0], hidden_dim
        )
        self.Q2 = QFunction(
            self.encoder.feature_dim, action_shape[0], hidden_dim
        )

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, obs, action, detach_encoder=False):
        # detach_encoder allows to stop gradient propogation to encoder
        obs = self.encoder(obs, detach=detach_encoder)

        q1 = self.Q1(obs, action)
        q2 = self.Q2(obs, action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2, obs

    def log(self, L, step, log_freq=LOG_FREQ):
        if step % log_freq != 0:
            return

        self.encoder.log(L, step, log_freq)

        for k, v in self.outputs.items():
            L.log_histogram('train_critic/%s_hist' % k, v, step)

        for i in range(3):
            L.log_param('train_critic/q1_fc%d' % i, self.Q1.trunk[i * 2], step)
            L.log_param('train_critic/q2_fc%d' % i, self.Q2.trunk[i * 2], step)


class SacAeAgent(object):
    """SAC+AE algorithm."""
    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        hidden_dim=256,
        discount=0.99,
        init_temperature=0.01,
        alpha_lr=1e-3,
        alpha_beta=0.9,
        actor_lr=1e-3,
        actor_beta=0.9,
        actor_log_std_min=-10,
        actor_log_std_max=2,
        actor_update_freq=2,
        critic_lr=1e-3,
        critic_beta=0.9,
        critic_tau=0.005,
        critic_target_update_freq=2,
        encoder_type='pixel',
        encoder_feature_dim=50,
        encoder_lr=1e-3,
        encoder_tau=0.005,
        decoder_type='pixel',
        decoder_lr=1e-3,
        decoder_update_freq=1,
        decoder_latent_lambda=0.0,
        decoder_weight_lambda=0.0,
        reward_size=1,
        num_layers=4,
        num_filters=32,
        model_state_dict=None,
        is_Finetune=False,
        is_Auxiliary=False,
        is_Transfer=False
    ):
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.decoder_update_freq = decoder_update_freq
        self.decoder_latent_lambda = decoder_latent_lambda
        self.is_Transfer = is_Transfer
        self.is_Auxiliary = is_Auxiliary
        self.is_Finetune = is_Finetune

        if self.is_Transfer:
            self.dynamics_model = DynamicsModel(encoder_feature_dim)
            self.dynamics_action_encoder = DynamicsActionEncoder(action_shape[0], hidden_dim, encoder_feature_dim)
            self.reward_model = RewardModel(encoder_feature_dim, reward_size)
            self.reward_action_encoder = RewardActionEncoder(action_shape[0], hidden_dim, encoder_feature_dim)

            self.dynamics_model.load_state_dict(model_state_dict['Dynamics'])
            self.reward_model.load_state_dict(model_state_dict['Reward'])
            self.dynamics_action_encoder.load_state_dict(model_state_dict['Dynamics_action_encode'])
            self.reward_action_encoder.load_state_dict(model_state_dict['Reward_action_encode'])

        if self.is_Auxiliary:
            self.dynamics_model = DynamicsModel(encoder_feature_dim)
            self.dynamics_action_encoder = DynamicsActionEncoder(action_shape[0], hidden_dim, encoder_feature_dim)
            self.reward_model = RewardModel(encoder_feature_dim, reward_size)
            self.reward_action_encoder = RewardActionEncoder(action_shape[0], hidden_dim, encoder_feature_dim)

        self.critic = Critic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters
        ).to(device)

        self.critic_target = Critic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters
        ).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        if self.is_Finetune:
            self.actor = Actor_Finetune(
                obs_shape, action_shape, hidden_dim, encoder_type,
                encoder_feature_dim, actor_log_std_min, actor_log_std_max,
                num_layers, num_filters
            ).to(device)
            self.actor.actor_head.load_state_dict(model_state_dict['Policy_head'])
            self.actor.encoder.copy_conv_weights_from(self.critic.encoder)
        else:
            self.actor = Actor(
                obs_shape, action_shape, hidden_dim, encoder_type,
                encoder_feature_dim, actor_log_std_min, actor_log_std_max,
                num_layers, num_filters
            ).to(device)
            self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)

        self.decoder = None
        if decoder_type != 'identity':
            # create decoder
            self.decoder = make_decoder(
                decoder_type, obs_shape, encoder_feature_dim, num_layers,
                num_filters
            ).to(device)
            self.decoder.apply(weight_init)

            # optimizer for critic encoder for reconstruction loss
            self.encoder_optimizer = torch.optim.Adam(
                self.critic.encoder.parameters(), lr=encoder_lr
            )

            # optimizer for decoder
            self.decoder_optimizer = torch.optim.Adam(
                self.decoder.parameters(),
                lr=decoder_lr,
                weight_decay=decoder_weight_lambda
            )

        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999)
        )

        if is_Auxiliary or is_Transfer:
            self.dynamics_model_optim = torch.optim.Adam([{'params': self.dynamics_model.parameters()},
                                              {'params': self.dynamics_action_encoder.parameters()}], lr=0.0003)
            self.reward_model_optim = torch.optim.Adam([{'params': self.reward_model.parameters()},
                                            {'params': self.reward_action_encoder.parameters()}], lr=0.0003)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        if self.decoder is not None:
            self.decoder.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, _, _, _ = self.actor(
                obs, compute_pi=False, compute_log_pi=False
            )
            return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, pi, _, _ = self.actor(obs, compute_log_pi=False)
            return pi.cpu().data.numpy().flatten()

    def update_critic(self, obs, action, reward, next_obs, not_done, L, step):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2, _ = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2, _ = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1,
                                 target_Q) + F.mse_loss(current_Q2, target_Q)
        L.log('train_critic/loss', critic_loss, step)


        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.log(L, step)

    def update_actor_and_alpha(self, obs, L, step):
        # detach encoder, so we don't update it with the actor loss
        _, pi, log_pi, log_std = self.actor(obs, detach_encoder=True)
        actor_Q1, actor_Q2, _ = self.critic(obs, pi, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        L.log('train_actor/loss', actor_loss, step)
        L.log('train_actor/target_entropy', self.target_entropy, step)
        entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)
                                            ) + log_std.sum(dim=-1)
        L.log('train_actor/entropy', entropy.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(L, step)

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_pi - self.target_entropy).detach()).mean()
        L.log('train_alpha/loss', alpha_loss, step)
        L.log('train_alpha/value', self.alpha, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update_decoder(self, obs, target_obs, action, next_obs, reward, L, step):
        h = self.critic.encoder(obs)

        if self.is_Auxiliary:
            with torch.no_grad():
                model_label = self.critic.encoder(next_obs)
            model_action_latent = self.dynamics_action_encoder(action)
            reward_action_latent = self.reward_action_encoder(action)
            model_pred = self.dynamics_model(h, model_action_latent)
            reward_pred = self.reward_model(h, reward_action_latent)
        elif self.is_Transfer:
            with torch.no_grad():
                model_label = self.critic.encoder(next_obs)
                model_action_latent = self.dynamics_action_encoder(action)
                reward_action_latent = self.reward_action_encoder(action)
                model_pred = self.dynamics_model(h, model_action_latent)
                reward_pred = self.reward_model(h, reward_action_latent)

        if target_obs.dim() == 4:
            # preprocess images to be in [-0.5, 0.5] range
            target_obs = utils.preprocess_obs(target_obs)
        rec_obs = self.decoder(h)
        rec_loss = F.mse_loss(target_obs, rec_obs)

        if self.is_Transfer or self.is_Auxiliary:
            model_loss = F.mse_loss(model_label, model_pred)
            reward_loss = F.mse_loss(reward, reward_pred)
        else:
            model_loss = 0
            reward_loss = 0

        # add L2 penalty on latent representation
        # see https://arxiv.org/pdf/1903.12436.pdf
        latent_loss = (0.5 * h.pow(2).sum(1)).mean()

        loss = rec_loss + self.decoder_latent_lambda * latent_loss + model_loss + reward_loss
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        L.log('train_ae/ae_loss', loss, step)

        self.decoder.log(L, step, log_freq=LOG_FREQ)

    def update(self, replay_buffer, L, step):
        obs, action, reward, next_obs, not_done = replay_buffer.sample()

        L.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.encoder, self.critic_target.encoder,
                self.encoder_tau
            )

        if self.decoder is not None and step % self.decoder_update_freq == 0:
            self.update_decoder(obs, obs, action, next_obs, reward, L, step)

    def save(self, model_dir, step):
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
        )
        if self.decoder is not None:
            torch.save(
                self.decoder.state_dict(),
                '%s/decoder_%s.pt' % (model_dir, step)
            )

    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step))
        )
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step))
        )
        if self.decoder is not None:
            self.decoder.load_state_dict(
                torch.load('%s/decoder_%s.pt' % (model_dir, step))
            )


    def train_model(self, replay_buffer, batch_size=256, holdout_ratio=0.2, max_epochs_since_update=5):

        state, action, reward, next_state, done = replay_buffer.sample_model(batch_size)

        self._max_epochs_since_update = max_epochs_since_update
        self._epochs_since_update = 0
        self._state = {}

        self._snapshots = (None, 1e10, 1e10)

        num_holdout = int(state.shape[0] * holdout_ratio)
        permutation = np.random.permutation(state.shape[0])
        state, action, reward, next_state, done = state[permutation], action[permutation], reward[permutation], next_state[permutation], done[permutation]

        train_state, train_action, train_reward, train_next_state, train_done = \
            state[num_holdout:], action[num_holdout:], reward[num_holdout:], next_state[num_holdout:], done[num_holdout:]
        holdout_state, holdout_action, holdout_reward, holdout_next_state, holdout_done = \
            state[num_holdout:], action[num_holdout:], reward[num_holdout:], next_state[num_holdout:], done[num_holdout:]

        holdout_state       = torch.from_numpy(holdout_state).float().to(self.device)
        holdout_action      = torch.from_numpy(holdout_action).float().to(self.device)
        holdout_reward      = torch.from_numpy(holdout_reward).float().to(self.device).unsqueeze(1)
        holdout_next_state  = torch.from_numpy(holdout_next_state).float().to(self.device)

        print('-------------------------Start Train-----------------------------')

        for epoch in itertools.count():
            train_idx = np.random.permutation(train_state.shape[0])
            for start_pos in range(0, train_state.shape[0], batch_size):
                # idx = train_idx[:, start_pos: start_pos + batch_size]
                idx = train_idx[start_pos: start_pos + batch_size]
                train_states = torch.from_numpy(train_state[idx]).float().to(self.device)
                train_actions = torch.from_numpy(train_action[idx]).float().to(self.device)
                train_rewards = torch.from_numpy(train_reward[idx]).float().to(self.device).unsqueeze(1)
                train_next_states = torch.from_numpy(train_next_state[idx]).float().to(self.device)

                with torch.no_grad():
                    feature_state_train = self.critic.encoder(train_states)
                    feature_next_state_train = self.critic.encoder(train_next_states)

                feature_dynamics_action = self.dynamics_action_encoder(train_actions)
                feature_reward_action   = self.reward_action_encoder(train_actions)

                losses = []

                # reward_pred  = self.reward_model(feature_state_train[0,:,:], feature_reward_action[0,:,:])
                # reward_label = train_rewards[0,:,:]
                reward_pred = self.reward_model(feature_state_train, feature_reward_action)
                reward_label = train_rewards
                reward_loss  = F.mse_loss(reward_pred, reward_label)

                state_pred = self.dynamics_model(feature_state_train, feature_dynamics_action)
                state_label = feature_next_state_train
                state_loss  = F.mse_loss(state_pred, state_label)

                self.dynamics_model_optim.zero_grad()
                state_loss.backward()
                self.dynamics_model_optim.step()

                self.reward_model_optim.zero_grad()
                reward_loss.backward()
                self.reward_model_optim.step()

                losses.append(state_loss)


            with torch.no_grad():
                feature_state_holdout           = self.critic.encoder(holdout_state)
                feature_next_state_holdout      = self.critic.encoder(holdout_next_state)
                feature_dynamics_action_holdout = self.dynamics_action_encoder(holdout_action)
                feature_reward_action_holdout   = self.reward_action_encoder(holdout_action)

                holdout_state_pred              = self.dynamics_model(feature_state_holdout, feature_dynamics_action_holdout)
                holdout_state_label             = feature_next_state_holdout
                holdout_state_mse_losses        = F.mse_loss(holdout_state_pred, holdout_state_label)
                holdout_state_mse_losses        = holdout_state_mse_losses.detach().cpu().numpy()

                holdout_reward_pred             = self.reward_model(feature_state_holdout, feature_reward_action_holdout)
                holdout_reward_label            = holdout_reward
                holdout_reward_mse_losses       = F.mse_loss(holdout_reward_pred, holdout_reward_label)
                holdout_reward_mse_losses       = holdout_reward_mse_losses.detach().cpu().numpy()

                break_train                     = self._save_best(epoch, holdout_state_mse_losses, holdout_reward_mse_losses)

                if break_train:
                    break
            print('epoch: {}, holdout state mse losses: {}, holdout reward mse losses: {}'.format(epoch,
                                                                                                  holdout_state_mse_losses,
                                                                                                      holdout_reward_mse_losses))

    def _save_best(self, epoch, holdout_state_losses, holdout_reward_losses):
        updated = False
        current_state_loss = holdout_state_losses
        current_reward_loss = holdout_reward_losses
        _, best_state, best_reward = self._snapshots
        improvement_state = (best_state - current_state_loss) / best_state
        improvement_reward = (best_reward - current_reward_loss) / best_reward

        if improvement_state > 0.01 or improvement_reward > 0.01:
            self._snapshots = (epoch, current_state_loss, current_reward_loss)
            updated = True

        if updated:
            self._epochs_since_update = 0
        else:
            self._epochs_since_update += 1
        if self._epochs_since_update > self._max_epochs_since_update:
            return True
        else:
            return False