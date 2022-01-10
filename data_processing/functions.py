# Name: Odysseas Papakyriakou
# Student_ID: 12632864

"""This module provides a number of functions to process data"""

import pandas as pd


def match_noc_regions(df1, df2):
    """Checks whether the noc regions in df1 correspond to the noc regions in df2.
    If there are regions that are represented differently in the two dataframes,
    it replaces the abbreviations so that they match.

    Parameters
    -----------
    df1 : the olympics dataframe
    df2 : the noc_regions datafrme

    Returns
    -----------
    df1 : the olympics dataframe that now has the same noc regions as the noc_regions dataframe
    df2 : the noc_regions dataframe that now has the same noc_regions as the olympics dataframe
    """

    # Create sets of the regions
    df1_regions = set(sorted(df1['NOC'].unique().tolist()))
    df2_regions = set(sorted(df2['NOC'].unique().tolist()))

    # Check if they have the same number of entries
    if len(df1_regions) == len(df2_regions):
        pass
    else:
        print('Length of unique NOC regions in df1:', len(df1_regions))
        print('Length of unique NOC regions in df2:', len(df2_regions))

    # Check if the entries match
    if len(df1_regions - df2_regions) != 0 and len(df2_regions - df1_regions) != 0:
        # NOC regions that are in df1, but not in df2
        not_in_df2 = df1_regions - df2_regions
        # NOC regions that are in df2, but not in df1
        not_in_df1 = df2_regions - df1_regions

        # After printing not_in_df2 and not_in_df1 we see that the only abbreviations that don't match are for
        # Singapore, so I replace the one NOC region with the other, so that all countries have the same abbreviation
        df2['NOC'] = df2['NOC'].replace('SIN', 'SGP')

        # Make sure the NOC regions now match
        df1_regions = set(sorted(df1['NOC'].unique().tolist()))
        df2_regions = set(sorted(df2['NOC'].unique().tolist()))
        if len(df1_regions - df2_regions) == 0 and len(df2_regions - df1_regions) == 0:
            pass
        else:
            print('Oops, NOC regions between the two dataframes are still not the same')
    else:
        print('The regions between the two dataframes are already identical.')

    return df1, df2


def merge_dfs(df1, df2):
    """Merges two dataframes on the 'NOC' column.

    Parameters
    -----------
    df1 : the olympics dataframe
    df2 : the noc_regions dataframe

    Returns
    -----------
    df : a dataframe that is merged on the 'NOC' column of the given dataframes"""

    merged_df = pd.merge(df1, df2, left_on='NOC', right_on='NOC')

    return merged_df


def read_extra_files(csv):
    """Reads and prepares the extra files required for the analysis.

    Parameters
    -----------
    csv : either of the two csv files:
          - 'population_total.csv
          - 'gdppercapita_us_inflation_adjusted.csv

    Returns
    -----------
    df : the csv file as a dataframe in a format that is best for further analysis
    """

    df = pd.read_csv(csv).set_index("country").transpose()
    df.index.name = "Year"

    return df
