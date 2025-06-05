# -*- coding: utf-8 -*-
"""
Created on Fri May 16 18:09:39 2025

@author: archit
"""

import itertools
import pandas as pd

# Define the values
T_vals = [300, 500, 800, 1000]
num_series_vals = [2, 5, 10, 25, 50, 100]
coint_frac_vals = [0.4, 0.6, 0.8]
num_of_rw = [0.4, 0.6]

# Cartesian product of all parameter values
all_combinations = list(itertools.product(T_vals, num_series_vals, coint_frac_vals, num_of_rw))

# Convert to DataFrame
df = pd.DataFrame(all_combinations, columns=['T', 'num_series', 'coint_frac', 'num_of_rw'])

# Optional: save to Excel
df.to_excel(r"C:\Users\archi\OneDrive\Desktop\VU\My VU\Thesis\Method\Simulation Study\parameter_combinations.xlsx", index=False)

# Show the first few rows
print(df.head())
