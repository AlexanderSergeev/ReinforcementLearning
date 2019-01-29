import matplotlib.pyplot as plt
import numpy as np

from mdp import FrozenLakeEnv
from mdp import has_graphviz



def draw_policy(mdp, state_values, fig=None):
    h, w = mdp.desc.shape
    states = sorted(mdp.get_all_states())
    V = np.array([state_values[s] for s in states])
    Pi = {s: get_optimal_action(mdp, state_values, s, gamma) for s in states}
    plt.imshow(V.reshape(w, h), cmap='gray', interpolation='none', clim=(0, 1))
    ax = plt.gca()
    ax.set_xticks(np.arange(h)-.5)
    ax.set_yticks(np.arange(w)-.5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    Y, X = np.mgrid[0:4, 0:4]
    a2uv = {'left': (-1, 0), 'down': (0, -1), 'right': (1,0), 'up': (-1, 0)}
    for y in range(h):
        for x in range(w):
            plt.text(x, y, str(mdp.desc[y,x].item()),
                     color='g', size=12,  verticalalignment='center',
                     horizontalalignment='center', fontweight='bold')
            a = Pi[y, x]
            if a is None:
                continue
            u, v = a2uv[a]
            plt.arrow(x, y,u*.3, -v*.3, color='m', head_width=0.1, head_length=0.1)
    plt.grid(color='b', lw=2, ls='-')
    plt.draw()
    plt.pause(2)
    if fig is not None:
        plt.cla()


def visualize_step_by_step(mdp, gamma, max_iter_number, min_difference):
    fig = plt.figure(figsize=(5, 5))
    state_values = {state: 0 for state in mdp.get_all_states()}
    for i in range(max_iter_number):
        new_state_values, done = rl_value_iteration(mdp, gamma, 1, min_difference, state_values)
        if done:
            break
        draw_policy(mdp, new_state_values, fig)
        state_values = new_state_values


def mass_gaming(mdp, gamma, num_iter, games_number, steps_number):
    state_values = {state: 0 for state in mdp.get_all_states()}
    state_values, _ = rl_value_iteration(mdp, gamma, num_iter, min_difference, state_values)

    total_rewards = []
    for game_i in range(games_number):
        s = mdp.reset()
        rewards = []
        for t in range(steps_number):
            # s, r, done, _ = mdp.step(get_optimal_action(mdp, state_values, s, gamma))
            rewards.append(r)
            if done:
                break
        total_rewards.append(np.sum(rewards))
    print('Average reward: ', np.mean(total_rewards))
    if mdp.slip_chance == 0:
        assert (1.0 <= np.mean(total_rewards) <= 1.0)
    else:
        assert (0.8 <= np.mean(total_rewards) <= 0.95)
    print('Well done!')


if __name__ == '__main__':
    visualize = True
    mdp = FrozenLakeEnv(map_name='8x8', slip_chance=0.1)
    mdp.render()

    gamma = 0.9
    num_iter = 100
    min_difference = 1e-5

    # Play in Frozen Lake Env
    state_values = {}  # Initialize state_values

    # Run value iteration algo!
    state_values, _ = None, None

    # See how our agent performs - e.g. render what is going on when agent choose `optimal` value
    s = mdp.reset()
    mdp.render()
    rewards = []  # Save all rewards to see mean reward.

    # Your code here!

    print('Average reward: ', np.mean(rewards))

    # if visualize:
    #     draw_policy(mdp, state_values)

    # Let's see how it is improving in time.
    # visualize_step_by_step(mdp, gamma, num_iter, min_difference)

    # Express test!
    mass_gaming(mdp, gamma, num_iter, 1000, 100)


