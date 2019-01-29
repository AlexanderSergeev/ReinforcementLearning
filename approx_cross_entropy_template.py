import gym
import gym.spaces
import time
import numpy as np


def generate_session(agent, t_max=10**5):
    states, actions = [], []
    total_reward = 0

    s = env.reset()

    for t in range(t_max):
        # Choose action from policy
        # You can use np.random.choice() func
        # a = ?
        a = None

        # Do action `a` to obtain new_state, reward, is_done,
        new_s, r, is_done = None, None, None

        # Record state, action and add up reward to states, actions and total_reward accordingly.
        # states
        # actions
        # total_reward

        # Update s for new cycle iteration

        if is_done:
            break

    return states, actions, total_reward


def select_elites(states_batch, actions_batch, rewards_batch, percentile=50):
    """
    Select states and actions from games that have rewards >= percentile
    :param states_batch: list of lists of states, states_batch[session_i][t]
    :param actions_batch: list of lists of actions, actions_batch[session_i][t]
    :param rewards_batch: list of rewards, rewards_batch[session_i][t]

    :returns: elite_states,elite_actions, both 1D lists of states and respective actions from elite sessions

    Please return elite states and actions in their original order
    [i.e. sorted by session number and timestep within session]

    If you're confused, see examples below. Please don't assume that states are integers (they'll get different later).
    """

    states_batch, actions_batch, rewards_batch = map(np.array, [states_batch, actions_batch, rewards_batch])

    # Compute reward threshold
    reward_threshold = None

    # Compute elite states using reward threshold
    elite_states = None

    # Compute elite actions using reward threshold
    elite_actions = None

    elite_states, elite_actions = map(np.concatenate, [elite_states, elite_actions])

    return elite_states, elite_actions


def rl_approx_cross_entropy(nn_agent):
    n_sessions = 100
    percentile = 70
    total_iterations = 100
    log = []

    for i in range(total_iterations):

        # Generate n_sessions for further analysis.
        sessions = None
        states_batch, actions_batch, rewards_batch = map(np.array, zip(*sessions))

        # Select elite states & actions.
        elite_states, elite_actions = None, None

        # Update policy using elite_states, elite_actions.
        # nn_agent

        # Info for debugging
        mean_reward = np.mean(rewards_batch)
        threshold = np.percentile(rewards_batch, percentile)
        log.append([mean_reward, threshold])

        print('Iteration= %.3f, Mean Reward = %.3f, Threshold=%.3f' % (i, mean_reward, threshold))

        if np.mean(rewards_batch) > 195:
            print('You Win! :)')
            break


def test_rl_approx_cross_entropy(nn_agent):
    s = env.reset()
    total_reward = 0
    for t in range(1000):
        # Choose action from nn_agent
        # You can use np.random.choice() func
        # a = ?
        a = None

        # Do action `a` to obtain new_state, reward, is_done,
        new_s, r, is_done = None, None, None

        if is_done:
            break
        else:
            env.render()
            time.sleep(0.07)
            total_reward += r
            # Update s for new cycle iteration

    print('Reward of Test agent = %.3f' % total_reward)


if __name__ == '__main__':
    # Create environment 'CartPole-v0'
    env = None
    s = env.reset()

    # Compute number of actions for this environment
    n_actions = None

    print('Actions number = %i' % n_actions)

    # Create neural network with 2 hidden layers of 10 & 10 neurons each & tanh activations
    # use MLPClassifier from scikit-learn

    agent = None

    # Initialize agent to the dimension of state and amount of actions
    agent.fit([s] * n_actions, range(n_actions))

    # Train `deep` neural network to approximate cross entropy method
    rl_approx_cross_entropy(agent)

    # Test our NN and see how it performs
    test_rl_approx_cross_entropy(agent)

    # Close environment when everything is done
    env.close()
