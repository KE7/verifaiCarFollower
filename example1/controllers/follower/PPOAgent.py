import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch import from_numpy, no_grad, save, load, tensor, clamp
from torch import float as torch_float
from torch import long as torch_long
from torch import min as torch_min
from torch.backends import mps
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np
from torch import manual_seed
from collections import namedtuple

Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state'])


class PPOAgent:
    """
    PPOAgent implements the PPO RL algorithm (https://arxiv.org/abs/1707.06347).
    It works with a set of discrete actions.
    It uses the Actor and Critic neural network classes defined below.
    """

    def __init__(self, image_width, image_height, image_channels, quantized_actions, clip_param=0.2, max_grad_norm=0.5, 
                 ppo_update_iters=5, batch_size=8, gamma=0.99, use_cuda=False, 
                 actor_lr=0.001, critic_lr=0.003, seed=None):
        super().__init__()
        if seed is not None:
            manual_seed(seed)

        # Hyper-parameters
        self.clip_param = clip_param
        self.max_grad_norm = max_grad_norm
        self.ppo_update_iters = ppo_update_iters
        self.batch_size = batch_size
        self.gamma = gamma
        self.use_cuda = use_cuda

        # models
        self.actor_net = Actor(image_width=image_width, image_height=image_height, in_channels=image_channels, quantized_actions=quantized_actions)
        self.critic_net = Critic(image_width=image_width, image_height=image_height, in_channels=image_channels)

        if self.use_cuda:
            self.actor_net.cuda()
            self.critic_net.cuda()
        elif mps.is_available():
            self.mps = True
            self.device = torch.device("mps")
            self.actor_net.to(device=self.device)
            self.critic_net.to(device=self.device)

        # Create the optimizers
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), actor_lr)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), critic_lr)

        # Training stats
        self.buffer = []

    def work(self, agentInput, type_="simple"):
        """
        type_ == "simple"
            Implementation for a simple forward pass.
        type_ == "selectAction"
            Implementation for the forward pass, that returns a selected action according to the probability
            distribution and its probability.
        type_ == "selectActionMax"
            Implementation for the forward pass, that returns the max selected action.
        """
        agentInput = from_numpy(np.array(agentInput)).float().unsqueeze(0)
        if self.use_cuda:
            agentInput = agentInput.cuda()
        elif self.mps:
            agentInput = agentInput.to(device=self.device)
        
        with no_grad():
            action_prob = self.actor_net(agentInput)

        if type_ == "simple":
            output = [action_prob[0][i].data.tolist() for i in range(len(action_prob[0]))]
            return output
        elif type_ == "selectAction":
            c = Categorical(action_prob)
            action = c.sample()
            return action.item(), action_prob[action.item()].item()
        elif type_ == "selectActionMax":
            return np.argmax(action_prob.cpu().data.numpy()).item(), 1.0
        else:
            raise Exception("Wrong type in agent.work(), returning input")

    def getValue(self, state):
        """
        Gets the value of the current state according to the critic model.

        :param state: agentInput
        :return: state's value
        """
        state = from_numpy(state)
        with no_grad():
            value = self.critic_net(state)
        return value.item()

    def save(self, path):
        """
        Save actor and critic models in the path provided.
        :param path: path to save the models
        :return: None
        """
        save(self.actor_net.state_dict(), path + '_actor.pkl')
        save(self.critic_net.state_dict(), path + '_critic.pkl')

    def load(self, path):
        """
        Load actor and critic models from the path provided.
        :param path: path where the models are saved
        :return: None
        """
        actor_state_dict = load(path + '_actor.pkl')
        critic_state_dict = load(path + '_critic.pkl')
        self.actor_net.load_state_dict(actor_state_dict)
        self.critic_net.load_state_dict(critic_state_dict)

    def storeTransition(self, transition):
        """
        Stores a transition in the buffer to be used later.

        :param transition: state, action, action_prob, reward, next_state
        :return: None
        """
        self.buffer.append(transition)

    def trainStep(self, batchSize=None):
        """
        Performs a training step or update for the actor and critic models, based on transitions gathered in the
        buffer. It then resets the buffer.
        If provided with a batchSize, this is used instead of default self.batch_size

        :param: batchSize: int
        :return: None
        """
        if batchSize is None:
            if len(self.buffer) < self.batch_size:
                return
            batchSize = self.batch_size

        # state = tensor([t.state for t in self.buffer], dtype=torch_float)
        state = from_numpy(np.array([t.state for t in self.buffer])).float()
        action = tensor([t.action for t in self.buffer], dtype=torch_long).view(-1, 1)
        reward = [t.reward for t in self.buffer]
        old_action_log_prob = tensor([t.a_log_prob for t in self.buffer], dtype=torch_float).view(-1, 1)

        # Unroll rewards
        R = 0
        Gt = []
        for r in reward[::-1]:
            R = r + self.gamma * R
            Gt.insert(0, R)
        Gt = tensor(Gt, dtype=torch_float)

        if self.use_cuda:
            state, action, old_action_log_prob = state.cuda(), action.cuda(), old_action_log_prob.cuda()
            Gt = Gt.cuda()
        elif self.mps:
            state, action, old_action_log_prob = state.to(device=self.device), action.to(device=self.device), old_action_log_prob.to(device=self.device)
            Gt = Gt.to(device=self.device)

        for _ in range(self.ppo_update_iters):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), batchSize, False):
                # Calculate the advantage at each step
                # print("Gt before shape is: ", Gt.shape)
                # Gt_index = Gt[index].view(-1, 1)
                Gt_index = Gt[index]
                # print("Gt after shape is: ", Gt_index.shape)
                V = self.critic_net(state[index])
                delta = Gt_index - V
                advantage = delta.detach()

                # Get the current prob
                action_prob = self.actor_net(state[index]).gather(0, action[index].squeeze())  # new policy

                # PPO
                ratio = (action_prob / old_action_log_prob[index])
                surr1 = ratio * advantage
                surr2 = clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

                # update actor network
                action_loss = -torch_min(surr1, surr2).mean()  # MAX->MIN descent
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # update critic network
                value_loss = F.mse_loss(Gt_index, V)
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()

        del self.buffer[:]


class Actor(nn.Module):
    def __init__(self, image_width, image_height, in_channels, quantized_actions):
        super(Actor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.maxPool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(59520, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.action_head = nn.Linear(512, quantized_actions)

    def forward(self, x):
        x = self.conv1(x)
        # print("first conv layer output shape: ", x.shape)
        x = self.conv2(x)
        # print("second conv layer output shape: ", x.shape)
        x = self.maxPool(x)
        # print("max pool layer output shape: ", x.shape)
        x = einops.rearrange(x, 'b c h w -> (b c h w)')
        # print("rearranged shape: ", x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_prob = F.softmax(self.action_head(x), dim=0)
        return action_prob


class Critic(nn.Module):
    def __init__(self, image_width, image_height, in_channels):
        super(Critic, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.maxPool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(59520, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.state_value = nn.Linear(512, 1)

    def forward(self, x):
        x = self.conv1(x)
        # print("first conv layer output shape: ", x.shape)
        x = self.conv2(x)
        # print("second conv layer output shape: ", x.shape)
        x = self.maxPool(x)
        # print("max pool layer output shape: ", x.shape)
        x = einops.rearrange(x, 'b c h w -> (b c h w)')
        # print("rearranged shape: ", x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.state_value(x)
        return value
