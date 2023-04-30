import numpy as np
import os

def plot_state_value_table(table, cols):
    for idx, state in enumerate(table):
        print(f"{table[state]:.2f}", end="\t")
        if (idx+1) % cols == 0:
            print()

def plot_q_table(table, cols):
    for r in range(5):
        for c in range(5):
            q_values = np.round(table[r][c],2)
            print(q_values, end="\t")
        print()

clear = lambda: os.system('cls' if os.name == 'nt' else 'clear')