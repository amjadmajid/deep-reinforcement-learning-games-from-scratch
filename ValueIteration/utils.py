import numpy as np
import os

def plot_state_value_table(state_value_table, num_cols):
    """
    This function prints a state value table with each value formatted to two decimal places.

    Parameters:
    state_value_table (dict): The state value table to be printed.
    num_cols (int): Number of columns for printing the table.
    """
    for idx, state in enumerate(state_value_table):
        print(f"{state_value_table[state]:.2f}", end="\t")
        if (idx+1) % num_cols == 0:
            print()

def plot_q_table(q_table):
    """
    This function prints a Q-table where each cell value is rounded to two decimal places.

    Parameters:
    q_table (list of list of float): The Q-table to be printed.
    """
    for row in q_table:
        for cell in row:
            q_values = np.round(cell, 2)
            print(q_values, end="\t")
        print()

def clear_screen():
    """
    This function clears the terminal screen, depending on the operating system.
    """
    os.system('cls' if os.name == 'nt' else 'clear')
