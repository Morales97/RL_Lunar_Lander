# EL2805 - Reinforcement Learning - Lab 2 Problem 1
# Solution by Valentin Minoz, Daniel Morales

import numpy as np
import gym
import torch
import matplotlib as mpl
from matplotlib import pyplot as plt
from tqdm import trange
from p1_agent import RandomAgent, DQNAgent, QAgent
from collections import deque


def running_average(x, N):
    ''' Function used to compute the running average
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y


def test_agent(env, agent, N, RENDER=False):
    ''' Let trained agent play N episodes'''
    env.reset()
    # Logging parameters
    n_ep_running_average = 50 # Running kernel length
    # For avg episodic reward and # of steps per episode
    episode_reward_list = []       # total reward per episode
    episode_number_of_steps = []   # number of steps per episode
    if hasattr(agent,'epsilon'): agent.epsilon = 0 # Full greedy mode in case
    ### Go
    EPISODES = trange(N, desc='', leave=True)
    for i in EPISODES:
        # Reset enviroment data and initialize variables
        done = False
        state = env.reset()
        total_episode_reward = 0.
        t = 0
        while not done:
            if RENDER: env.render()
            # Take an action
            action = agent.forward(state)
            # Get next state, reward, done
            next_state, reward, done, _ = env.step(action)
            # Update episode reward
            total_episode_reward += reward
            # Update state for next iteration
            state = next_state
            t += 1
        # Append episode reward and total number of steps
        episode_reward_list.append(total_episode_reward)
        episode_number_of_steps.append(t)
        # Close environment
        env.close()
    return N, episode_reward_list, episode_number_of_steps


def run_DQN(env, N_episodes = 300, discount_factor = .99,
            size_exprb = 10000, nominal_score = False):
    ''' Train and return a DQNAgent and training stats'''
    env.reset()
    # Constants
    n_actions = env.action_space.n              # Number of available actions
    dim_state = len(env.observation_space.high) # State dimensionality

    # Learning parameters
    learning_rate = 5*1e-4 # "alpha"
    size_batch = 50 # "N"          # Training batch size
    target_net_update_freq = size_exprb // size_batch # "C"
    # ε
    epsilon_max = 0.99
    epsilon_min = 0.05
    epsilon_tau = 0.90 # For what initial ratio of i:s does eps decay
    print(f'\ndiscount γ: {discount_factor}')
    print(f'N episodes: {N_episodes}')
    print(f'learning r: {learning_rate}')
    print(f'sz expeRB:  {size_exprb}')
    print(f'sz batch:   {size_batch}')
    print(f'C:          {target_net_update_freq}')
    print(f'\neps min:    {epsilon_max}')
    print(f'eps max:    {epsilon_min}')
    print(f'eps time:   {epsilon_tau}')

    # Sanity
    assert type(N_episodes) is int and N_episodes > 0
    assert discount_factor <= 1 and discount_factor >= 0
    assert epsilon_max > epsilon_min
    assert type(size_exprb) is int and size_exprb > 0
    assert type(size_batch) is int and size_batch > 0

    # Implicit parameters
    decay_episodes = int(epsilon_tau * N_episodes)
    lin_decay_rate = (epsilon_min - epsilon_max) / decay_episodes
    exp_decay_rate = (epsilon_min / epsilon_max)** (1/decay_episodes)

    # Logging variables
    n_ep_running_average = 50 # Running avg kernel length
    episode_reward_list = []       # total reward per episode
    episode_number_of_steps = []   # number of steps per episode

    # Experience buffer initialization
    ERB = deque(maxlen = size_exprb)

    # DQN agent initialization
    agent = DQNAgent(n_actions, dim_state,
                    discount_factor, epsilon_max,
                    learning_rate)

    ### Training process
    EPISODES = trange(N_episodes, desc='', leave=True)
    tot_its = 0
    epss = []
    for i in EPISODES:
        epss.append(agent.epsilon)
        # Reset enviroment data and initialize variables
        done = False
        state = env.reset()
        total_episode_reward = 0.
        t = 0
        while not done:
            # Take an action
            action = agent.forward(state)
            assert type(action) == int, action

            # Get next state, reward, done and append to buffer
            next_state, reward, done, _ = env.step(action)
            ERB.append((state, action, reward, next_state, done))

            ## Perform Training ##
            if len(ERB) >= size_batch:
                # get random batch
                batch_idxs = np.random.choice(
                    len(ERB),
                    size = size_batch,
                    replace = False)
                batch = [ERB[index] for index in batch_idxs]
                agent.train(batch)# train on batch

            # Update target network
            if not tot_its%target_net_update_freq:
                agent.update_target()

            # Update episode reward
            total_episode_reward += reward
            # Update state for next iteration
            state = next_state
            t += 1
            tot_its += 1

        # Update epsilon
        if i < decay_episodes:
            # Linear eps decay:
            # agent.epsilon += lin_decay_rate
            # Exponential eps decay:
            agent.epsilon *= exp_decay_rate
        if i > decay_episodes:
            assert abs(agent.epsilon-epsilon_min) < 1e-4

        # Append episode reward and total number of steps
        episode_reward_list.append(total_episode_reward)
        episode_number_of_steps.append(t)
        # Close environment
        env.close()

        if not i%max([10,(N_episodes//100)]):
            # Updates the tqdm update bar with fresh information
            # (episode number, total reward of the last episode, total number of Steps
            # of the last episode, average reward, average number of steps)
            last_ravr = running_average(episode_reward_list, n_ep_running_average)[-1]
            EPISODES.set_description(
                f"R:{total_episode_reward:.1f}(avg:{last_ravr:.1f}); T:{t} "+
                "(avg:{})".format(
                running_average(episode_number_of_steps, n_ep_running_average)[-1]))
            if nominal_score and last_ravr > nominal_score:
                print('Average R good. Ending training')
                N_episodes = i+1
                break
    # plt.figure()
    # plt.plot(epss)
    # plt.show()
    return agent, N_episodes, episode_reward_list, episode_number_of_steps


def plot_rewards(N, ep_rewards, ep_steps, ker_length,
                 showplot=True, filename = False):
    ''' Plot rewards and steps '''
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
    ax[0].plot([i for i in range(1, N+1)], ep_rewards, label='Episode reward')
    ax[0].plot([i for i in range(1, N+1)], running_average(
        ep_rewards, ker_length), label='Avg. episode reward')
    ax[0].set_ylim(-500, 500)
    ax[0].set_xlabel('Episodes')
    ax[0].set_ylabel('Total reward')
    ax[0].set_title('Total Reward vs Episodes')
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    ax[1].plot([i for i in range(1, N+1)], ep_steps, label='Steps per episode')
    ax[1].plot([i for i in range(1, N+1)], running_average(
        ep_steps, ker_length), label='Avg. number of steps per episode')
    # ax[1].set_ylim(-10, 500)
    ax[1].set_xlabel('Episodes')
    ax[1].set_ylabel('Total number of steps')
    ax[1].set_title('Total number of steps vs Episodes')
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    if showplot: plt.show()
    if filename: plt.savefig(filename)
    plt.close(fig)


def discount_study(env, discounts):
    ''' Part (e)(2)'''
    for d in discounts:
        _, N, ep_rew, ep_steps = run_DQN(env, discount_factor = d)
        name = f'discount-{d}.png'
        plot_rewards(N, ep_rew, ep_steps, 50, showplot = False, filename = name)


def buffer_size_study(env, mem_sizes):
    ''' Part (e)(3)'''
    for b in mem_sizes:
        _, N, ep_rew, ep_steps = run_DQN(env, size_exprb = b)
        name = f'size_exprb-{b}.png'
        plot_rewards(N, ep_rew, ep_steps, 50, showplot = False, filename = name)


def val_func_plot(nnfilename = 'neural-network-1.pth', do_3d_surf=False):
    ''' Part (f)'''
    n_y = 100
    n_om = 100
    ys = np.linspace(0, 1.5, n_y)
    omegas = np.linspace(-np.pi, np.pi, n_om)
    Om, Y = np.meshgrid(omegas, ys)
    states = np.array(
        [[(0,y,0,0,omega,0,0,0) for omega in omegas] for y in ys],
        dtype=np.float32)
    Q = torch.load(nnfilename)
    with torch.no_grad():
        max_and_argmax =[[torch.max(Q(torch.tensor([states[i,j]])), axis=1)
            for j in range(n_om)] for i in range(n_y)]
    np.array(max_and_argmax).shape
    V = np.array(
        [[max_and_argmax[i][j][0].item() for j in range(n_om)] for i in range(n_y)])
    a_star = np.array(
        [[max_and_argmax[i][j][1].item() for j in range(n_om)] for i in range(n_y)])

    fig = plt.figure(figsize=(9,9))
    if do_3d_surf:
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(Om, Y, V, cmap=mpl.cm.coolwarm)
        ax.set_ylabel('y')
        ax.set_xlabel('ω')
        ax.set_zlabel('V(s(y,ω))')
    else:
        ax = fig.add_subplot(111)
        im = ax.pcolormesh(Om, Y, V, cmap=mpl.cm.coolwarm)
        fig.colorbar(im)
        ax.set_ylabel('y')
        ax.set_xlabel('ω')
        ax.set_title('V(s(y,ω))')
    plt.savefig('Qmax.png')
    plt.close(fig)

    fig = plt.figure(figsize=(9,9))
    ax = fig.add_subplot(111)
    im = ax.pcolormesh(Om, Y, a_star, cmap=mpl.cm.viridis)
    cbar = fig.colorbar(im, ticks=[0,1,2,3])
    cbar.ax.set_yticklabels(['nothing (0)','left (1)','main (2)','right (3)'])
    ax.set_ylabel('y')
    ax.set_xlabel('ω')
    ax.set_title('π(s(y,ω))')
    plt.savefig('Qargmax.png')
    plt.close(fig)


def comparison_to_random(env, nnfilename = 'neural-network-1.pth'):
    ''' Part (g) '''
    N = 50
    random_agent = RandomAgent(env.action_space.n)
    rn, random_a_rewards, _ = test_agent(env, random_agent, N)
    dqn_agent = QAgent(nnfilename)
    dqnn, dqn_a_rewards, _ = test_agent(env, dqn_agent, N)
    assert (rn == N and dqnn == N)
    # Plot rewads
    fig = plt.figure(figsize=(9, 9))
    plt.plot([i for i in range(1, N+1)], random_a_rewards,
             'ro-', label='Random Agent')
    plt.plot([i for i in range(1, N+1)], dqn_a_rewards,
             'bo-', label='Trained DQN Agent')
    plt.ylim(-500, 500)
    plt.xlabel('Episodes')
    plt.ylabel('Total reward')
    plt.title('Total Reward vs Episodes')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('DQNvsRandom.png')
    plt.close(fig)


def main():
    cases = ['base',
    'discount_study',
    'N_episodes_study',
    'buffer_size_study',
    'val_func_plot',
    'comparison_to_random']

    ## USER
    case = input(f'Cases:\n{cases}\nInput index:\n')
    try:
        case = int(case)
        print(f'Running case [{case}]: {cases[case]}')
    except: print('INVALID')

    # Get the discrete Lunar Lander Environment
    env = gym.make('LunarLander-v2')

    ## RUN WHATEVER IS CHOSEN
    if case == 0:
        # (c) Do run_DQN with default args - gives good results
        agent, N, ep_rew, ep_steps = run_DQN(env)
        plot_rewards(N, ep_rew, ep_steps, 50)
        if (running_average(ep_rew, 50)[-1] > 50 and
            input('Input Anything to Save or Enter to Exit...')):
            agent.save_network()

    # (e)
    elif case == 1: # (2) discount
        discount_study(env, [.80,.99, 1]) # [lower,default,higher]
    elif case == 2: # (3).1 Nepisodes
        # Same as case[0] but some other N
        N_eps = 1200
        agent, N, ep_rew, ep_steps = run_DQN(env, N_episodes = N_eps)
        plot_rewards(N, ep_rew, ep_steps, 50, filename = f'{N_eps}.png')
        assert N == N_eps
        agent.save_network('net{N_eps}.pth')
    elif case == 3: # (3).2 ERBsize
        buffer_size_study(env, [5000,10000,50000]) # [smaller,default,larger]

    elif case == 4: # (f)
        val_func_plot()

    elif case == 5: # (g)
        comparison_to_random(env)


if __name__ == '__main__': main()
