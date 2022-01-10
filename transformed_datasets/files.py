# Name: Odysseas Papakyriakou
# Student_ID: 12632864

"""This module opens the transformed files, so they can be used in other parts of the project."""

import pandas as pd

olympics = pd.read_csv('Transformed Data/data.csv')

gdp = pd.read_csv('Transformed Data/gdp.csv', index_col='Year')

with open('Transformed Data/countries.csv', 'r') as f:
    countries = [c.strip() for c in f]

final = pd.read_csv('Transformed Data/final.csv')

pop = pd.read_csv('Transformed Data/pop.csv')

