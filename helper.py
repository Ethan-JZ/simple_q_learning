import pandas as pd
import numpy as np
from typing import List, Tuple, Union
import time


def build_q_table(n_states: int, actions: List[str]) -> pd.DataFrame:
    """
    build q table based on state and actions
    """

    table = pd.DataFrame(
        np.zeros((n_states, len(actions))), # initial values for q table
        columns=actions, # action names
    )

    print(table)

    return table

def choose_action(state: int, actions: List[str], q_table: pd.DataFrame, epsilon: float) -> str:

    """
    This function tells how to choose an action
    state: current state index
    actions: action choices
    q_table: action value table with each q(s, a) pair
    epsilon: epsion greedy parameter
    """
    state_actions = q_table.iloc[state, :]  # choose the entire row q(s, a) of the state s
    
    # act in exploring way if random number is larger than epsilon or all q(s, a) = 0, when initialize the exploration
    if np.random.uniform() > epsilon or state_actions.all() == 0:
        action_name = np.random.choice(actions)
    
    else: # act in greedy way, only choose the one with largest q(s, a)
        action_name = state_actions.idxmax()
    return action_name


def get_env_feedback(state: int, action: str, n_states: int) -> Tuple[Union[int, str], int]:
    """
    This is how agent will interact with the environment
    state: current state index
    action: action to take for the agent
    n_states: number of states

    return: next state index, reward of the current action
    """
    if action == 'right': # move right
        if state == n_states - 2:  # terminate
            state_next = 'terminal'
            reward = 1
        else:
            state_next = state + 1
            reward = 0

    else:  # move left
        reward = 0

        if state == 0:  # hit the wall, stay at current state
            state_next = state  # 
        else:
            state_next = state - 1
    
    return state_next, reward


def update_env(state: int, episode: int, step_counter: int, n_states: int, fresh_time: float) -> None:
    """
    This is how environment should be updated: '-----T', T is our target
    state: current state index
    episode: current episode number
    step_counter: how many steps are there after each episode
    n_states: number of states
    fresh_time: fresh time per movement
    """
    
    env = ['⬜️'] * (n_states - 1) + ['⭐️']
    if state == 'terminal':
        
        interaction = 'Episode %s: total steps = %s' % (episode+1, step_counter)
        print(interaction)
        time.sleep(2)
    
    else:
        env[state] = '👻'  # mark the current agent position
        interaction = ''.join(env)
        print(interaction)
        time.sleep(fresh_time)

