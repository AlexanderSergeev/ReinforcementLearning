import gym
import gym.spaces
import gym.wrappers
import numpy as np

import torch, torch.nn as nn
from torch.autograd import Variable

from tensorboardX import SummaryWriter
import datetime


def test_define_network(environment, net):
    s = environment.reset()
    assert tuple(net(Variable(torch.FloatTensor([s]*3))).size()) == (3, n_actions), \
        'please make sure your model maps state s -> [Q(s,a0), ..., Q(s, a_last)]'

    assert isinstance(list(net.modules())[-1], nn.Linear), \
        'please make sure you predict q-values without nonlinearity (ignore if you know what you are doing)'
    assert isinstance(get_action(s), int), \
        'get_action(s) must return int, not %s. try int(action)' % (type(get_action(s)))

    print('Test #1: define_network() & get_action() functions: OK!')


def test_eps_greedy_strategy():
    # Test epsilon-greedy exploration
    for eps in [0., 0.1, 0.5, 1.0]:
        state_frequencies = np.bincount([get_action(s, epsilon=eps) for i in range(10000)], minlength=n_actions)
        best_action = state_frequencies.argmax()
        assert abs(state_frequencies[best_action] - 10000 * (1 - eps + eps / n_actions)) < 200
        for other_action in range(n_actions):
            if other_action != best_action:
                assert abs(state_frequencies[other_action] - 10000 * (eps / n_actions)) < 200
        print('eps=%.1f tests passed' % eps)
    print('Test #2: epsilon greedy exploration: OK!')


def test_td_loss(environment, net):
    s = environment.reset()
    a = environment.action_space.sample()
    next_s, r, done, _ = env.step(a)
    loss = compute_td_loss([s], [a], [r], [next_s], [done], check_shapes=False)
    loss.backward()

    assert isinstance(loss, Variable) and tuple(loss.data.size()) == (1,), \
        'you must return scalar loss - mean over batch'
    assert np.any(next(net.parameters()).grad.data.numpy() != 0), \
        'loss must be differentiable w.r.t. network weights'

    print('Test #3: compute_td_loss() function: OK!')


def to_one_hot(y, n_dims=None):
    """ helper #1: take an integer vector (tensor of variable) and convert it to 1-hot matrix. """
    y_tensor = y.data if isinstance(y, Variable) else y
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    return Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot


def where(cond, x_1, x_2):
    """ helper #2: like np.where but in PyTorch. """
    return (cond * x_1) + ((1-cond) * x_2)


# < YOUR CODE HERE >
def define_network(state_dim, n_actions):
    network = nn.Sequential()
    return network


# < YOUR CODE HERE >
def get_action(state, epsilon=0):
    """
    sample actions with epsilon-greedy policy
    recap: with probability = epsilon pick random action, else pick action with highest Q(s,a)
    """
    state = None
    q_values = None

    return None


# < YOUR CODE HERE >
def compute_td_loss(states, actions, rewards, next_states, is_done, gamma=0.99, check_shapes=False):
    """ Compute td loss using torch operations only."""
    states = Variable(torch.FloatTensor(states))  # shape: [batch_size, state_size]
    actions = Variable(torch.IntTensor(actions))  # shape: [batch_size]
    rewards = Variable(torch.FloatTensor(rewards))  # shape: [batch_size]
    next_states = Variable(torch.FloatTensor(next_states))  # shape: [batch_size, state_size]
    is_done = Variable(torch.FloatTensor(is_done))  # shape: [batch_size]

    # get q-values for all actions in current states
    predicted_qvalues = None  # < YOUR CODE HERE >

    # select q-values for chosen actions
    predicted_qvalues_for_actions = torch.sum(predicted_qvalues.cpu() * to_one_hot(actions, n_actions), dim=1)

    # compute q-values for all actions in next states
    predicted_next_qvalues = None  # < YOUR CODE HERE >

    # compute V*(next_states) using predicted next q-values
    next_state_values = None  # < YOUR CODE HERE >

    assert isinstance(next_state_values.data, torch.FloatTensor)

    # compute 'target q-values' for loss
    target_qvalues_for_actions = None  # < YOUR CODE HERE >

    # at the last state we shall use simplified formula: Q(s,a) = r(s,a) since s' doesn't exist
    target_qvalues_for_actions = where(is_done, rewards, target_qvalues_for_actions).cpu()

    # Mean Squared Error loss to minimize
    loss = torch.mean((predicted_qvalues_for_actions - target_qvalues_for_actions.detach()) ** 2)

    if check_shapes:
        assert predicted_next_qvalues.data.dim() == 2, \
            'make sure you predicted q-values for all actions in next state'
        assert next_state_values.data.dim() == 1, \
            'make sure you computed V(s-prime) as maximum over just the actions axis and not all axes'
        assert target_qvalues_for_actions.data.dim() == 1, \
            'there is something wrong with target q-values, they must be a vector'

    return loss


def generate_session(t_max=1000, epsilon=0, train=False):
    """Play env with approximate q-learning agent and train it at the same time"""
    total_reward = 0
    s = env.reset()

    for t in range(t_max):
        # a = <get_action_a> from agent # < YOUR CODE HERE >
        a = None
        next_s, r, done, _ = env.step(a)
        if train:
            opt.zero_grad()
            loss = compute_td_loss([s], [a], [r], [next_s], [done])
            loss.backward()
            opt.step()

        total_reward += r
        s = next_s
        if done:
            break

    return total_reward


if __name__ == '__main__':
    dump_logs = False
    record_video = False
    env = gym.make("CartPole-v0").env
    s = env.reset()
    n_actions = env.action_space.n
    state_dim = env.observation_space.shape
    n_actions = env.action_space.n

    print('Actions number = %i , State example = %s ' % (n_actions, s))
    print('State space upper bound: %s' % env.observation_space.high)
    print('State space lower bound: %s' % env.observation_space.low)

    # Complete define_network() & get_action() functions
    network = define_network(state_dim, n_actions)

    test_define_network(env, network)
    test_eps_greedy_strategy()

    # Complete compute_td_loss function
    test_td_loss(env, network)

    # Create Adam optimizer with lr=1e-4
    opt = torch.optim.Adam(network.parameters(), lr=1e-4)
    epsilon = 0.4
    max_epochs = 1000
    if dump_logs:
        log_path = './logs/{:%Y_%m_%d_%H_%M}'.format(datetime.datetime.now())
        writer = SummaryWriter(log_path)

    for i in range(max_epochs):
        session_rewards = [generate_session(epsilon=epsilon, train=True) for _ in range(100)]
        print('Epoch #{}\tMean reward = {:.3f}\tEpsilon = {:.3f}'.format(i, np.mean(session_rewards), epsilon))
        if dump_logs:
            writer.add_scalar('Mean Reward', np.mean(session_rewards), i)

        # Code Epsilon decay <HERE>
        # epsilon ?
        assert epsilon >= 1e-4, 'Make sure epsilon is always nonzero during training'

        if np.mean(session_rewards) > 300:
            print('You Win!')
            break
    if record_video:
        env = gym.wrappers.Monitor(gym.make('CartPole-v0').env, directory='videos', force=True)
        sessions = [generate_session(epsilon=0, train=False) for _ in range(100)]
    env.close()