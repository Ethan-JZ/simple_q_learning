import numpy as np
from helper import build_q_table, choose_action, get_env_feedback, update_env
import matplotlib.pyplot as plt


def q_learning():

    ######### Global variables ##########
    np.random.seed(2)

    N_STATES = 8                 # the length of the 1 dim world
    ACTIONS = ['left', 'right']  # available actions
    EPSILON = 0.9                # epsilon greedy parameter
    ALPHA = 0.1                  # learning rate
    LAMBDA = 0.9                 # discounted factor
    MAX_EPISODES = 15            # maximum episodes
    FRESH_TIME = 0.01            # fresh time per move of the agent
    
    # init the action values table
    steps_per_episode = []
    q_table = build_q_table(n_states=N_STATES, actions=ACTIONS)

    # start the loop for all episodes
    for episode in range(MAX_EPISODES):
        
        # initialize the environment and agent
        step_counter = 0
        state = 0
        done = False
        update_env(state=state, episode=episode, step_counter=step_counter, n_states=N_STATES, fresh_time=FRESH_TIME)
        
        # loop for each episode
        while not done:
            action = choose_action(state=state, actions=ACTIONS, q_table=q_table, epsilon=EPSILON)
            state_next, reward = get_env_feedback(state=state, action=action, n_states=N_STATES)  # take action and get to the next state
            q_estimate = q_table.loc[state, action]
            
            # if next state is not terminal
            if state_next != 'terminal':
                q_target = reward + LAMBDA * q_table.iloc[state_next, :].max()
            
            # if next state is terminal
            else:
                q_target = reward
                done = True
        
            # update q table
            q_table.loc[state, action] += np.round(ALPHA * (q_target - q_estimate), 4)
            state = state_next
            
            # update environment and the agent
            update_env(state=state, episode=episode, step_counter=step_counter, n_states=N_STATES, fresh_time=FRESH_TIME)
            step_counter += 1
        
        # append the steps for this episode
        steps_per_episode.append(step_counter)
    
    return q_table, steps_per_episode


if __name__ == "__main__":
    q_table, steps_per_episode = q_learning()
    print('\r\n Q-table: \n')
    print(q_table)

    # Plot episode vs steps
    plt.figure()
    plt.plot(range(1, len(steps_per_episode)+1), steps_per_episode, marker='o')
    plt.xlabel('Episode')
    plt.ylabel('Total Steps')
    plt.title('Q-learning: Episode vs Steps')
    plt.grid(True)
    plt.show()
