from mdp import MDP
import matplotlib.pyplot as plt
import numpy as np


def check_generate_session_func(mdp, get_action_value_func):
    test_Vs = {s: i for i, s in enumerate(sorted(mdp.get_all_states()))}
    assert np.allclose(get_action_value_func(mdp, test_Vs, 's2', 'a1', 0.9), 0.69)
    assert np.allclose(get_action_value_func(mdp, test_Vs, 's1', 'a0', 0.9), 3.95)

    print('get_action_value() function: Ok!')


def check_get_new_state_value_func(mdp, get_new_state_value_func):
    test_Vs = {s: i for i, s in enumerate(sorted(mdp.get_all_states()))}
    test_Vs_copy = dict(test_Vs)
    assert np.allclose(get_new_state_value_func(mdp, test_Vs, 's0', 0.9), 1.8)
    assert np.allclose(get_new_state_value_func(mdp, test_Vs, 's2', 0.9), 0.69)
    assert test_Vs == test_Vs_copy, 'please do not change state_values in get_new_state_value'

    print('get_new_state_value() function: Ok!')


def check_state_values(state_values):
    assert abs(state_values['s0'] - 8.032) < 0.01
    assert abs(state_values['s1'] - 11.169) < 0.01
    assert abs(state_values['s2'] - 8.921) < 0.01

    print('Checking final state_values: Ok!')


def check_get_optimal_action(get_optimal_action_func, mdp, state_values, gamma):
    assert get_optimal_action_func(mdp, state_values, 's0', gamma) == 'a1'
    assert get_optimal_action_func(mdp, state_values, 's1', gamma) == 'a0'
    assert get_optimal_action_func(mdp, state_values, 's2', gamma) == 'a0'

    print('get_optimal_action() function : Ok!')


def get_action_value(mdp, state_values, state, action, gamma):
    """ Computes Q(s,a) """
    q = 0

    return q


def get_new_state_value(mdp, state_values, state, gamma):
    """ Computes next V(s) .Please do not change state_values in process. """
    if mdp.is_terminal(state):
        return 0

    return 0


def rl_value_iteration(mdp, gamma, num_iter, min_difference, init_state_values):
    # Initialize V(s)
    state_values = init_state_values

    for i in range(num_iter):
        # Compute new state values using the functions you defined above.
        # It must be a dict {state : float V_new(state)}
        new_state_values = {}

        assert isinstance(new_state_values, dict)

        # Compute difference
        diff = max(abs(new_state_values[s] - state_values[s]) for s in mdp.get_all_states())
        print('Iteration = %4i | Difference = %.3f |   ' % (i, diff), end='')
        print('   '.join('V(%s) = %.3f' % (s, v) for s, v in state_values.items()), end='\n')

        # Updating state_values
        state_values = None

        if diff < min_difference:
            print('Done!')
            return state_values, True

    return state_values, False


def get_optimal_action(mdp, state_values, state, gamma=0.9):
    """ Finds optimal action. """
    if mdp.is_terminal(state):
        return None

    return None


def test_optimal_strategy(mdp, state_values, gamma, max_steps):
    """ Test optimal strategy, derived from state_values. """
    rewards_at_each_step = []
    s = mdp.reset()
    for _ in range(max_steps):
        s, r, done, _ = mdp.step(get_optimal_action(mdp, state_values, s, gamma))
        rewards_at_each_step.append(r)

    return rewards_at_each_step


if __name__ == '__main__':
    transition_probs = {
        's0': {
            'a0': {'s0': 0.5, 's2': 0.5},
            'a1': {'s2': 1}
        },
        's1': {
            'a0': {'s0': 0.7, 's1': 0.1, 's2': 0.2},
            'a1': {'s1': 0.95, 's2': 0.05}
        },
        's2': {
            'a0': {'s0': 0.4, 's1': 0.6},
            'a1': {'s0': 0.3, 's1': 0.3, 's2': 0.4}
        }
    }
    rewards = {
        's1': {'a0': {'s0': +5}},
        's2': {'a1': {'s0': -1}}
    }

    gamma = 0.9  # Discount factor for MDP

    mdp = MDP(transition_probs, rewards, initial_state='s0')

    print('Initial state =', mdp.reset())
    next_state, reward, done, info = mdp.step('a1')
    print('Next_state = %s, reward = %s, done = %s' % (next_state, reward, done))

    print('mdp.get_all_states = ', mdp.get_all_states())
    print("mdp.get_possible_actions('s1') = ", mdp.get_possible_actions('s1'))
    print("mdp.get_next_states('s1', 'a0') = ", mdp.get_next_states('s1', 'a0'))
    print("mdp.get_reward('s1', 'a0', 's0') = ", mdp.get_reward('s1', 'a0', 's0'))
    print("mdp.get_transition_prob('s1', 'a0', 's0') = ", mdp.get_transition_prob('s1', 'a0', 's0'))

    visualize = False
    from mdp import has_graphviz
    print('Graphviz available: ', has_graphviz)

    if has_graphviz and visualize:
        from mdp import plot_graph, plot_graph_with_state_values, plot_graph_optimal_strategy_and_state_values
        plot_graph(mdp).render()

    # Complete get_action_value().
    check_generate_session_func(mdp, get_action_value)

    # Complete get_new_state_value()
    check_get_new_state_value_func(mdp, get_new_state_value)

    # Let's combine everything together

    # Complete rl_value_iteration()

    # Test rl_value_iteration()
    num_iter = 100  # Maximum iterations, excluding initialization
    min_difference = 0.001  # stop Value Iteration if new values are this close to old values (or closer)

    init_values = {}
    state_values, _ = rl_value_iteration(mdp, gamma, num_iter, min_difference, init_values)

    # Draw state_values after training.
    if has_graphviz and visualize:
        plot_graph_with_state_values(mdp, state_values).render(filename='MDP_with_states')

    print('Final state values:', state_values)
    check_state_values(state_values)

    # Complete get_optimal_action function.
    check_get_optimal_action(get_optimal_action, mdp, state_values, gamma)

    # Visualize optimal strategy.
    if has_graphviz and visualize:
        plot_graph_optimal_strategy_and_state_values(mdp, state_values, get_action_value, gamma).render(
            filename='MDP_with_opt_strategy')

    print([get_optimal_action(mdp, state_values, s, gamma=0.9) for s in mdp.get_all_states()])
    # Test optimal strategy.
    rewards = test_optimal_strategy(mdp, state_values, gamma, 10000)
    print('Average reward: ', np.mean(rewards))
    assert (0.85 < np.mean(rewards) < 1.0)