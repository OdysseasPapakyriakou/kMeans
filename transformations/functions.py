# Name: Odysseas Papakyriakou
# Student_ID: 12632864

"""This module provides a number of functions to transform the data for the project"""

import pandas as pd
from functools import reduce


def match_countries(df_to_match, olympics):
    """Changes the names of the countries in the df_to_match df so that they match
    the names of the countries in the olympics df.

    Parameters
    -----------
    df_to_match : either of the two dataframes:
                  - gdp
                  - pop
    olympics    : the olympics dataframe

    Returns
    -----------
    df_to_match      : the dataframe given as first parameter that now its countries
                       match the countries in the olympics df
    common_countries : a list with the common countries in the two dataframes
    """

    # countries in the to_match df
    df_countries = set(df_to_match.columns.tolist())
    # countries in the olympics df
    ol_regions = set(sorted(olympics.region.unique().tolist()))

    # countries in the to_match df that are not in the olympics df
    not_in_ol = df_countries.difference(ol_regions)
    # countries in the olympics df that are not in the to_match df
    not_in_df = ol_regions.difference(df_countries)

    # After printing not_in_il and not_int_df, we see that some countries are simply named differently
    # Therefore, I renames these countries in the to_match df so that they match the countries from the olympics df
    df_to_match.rename(columns={"United States": "USA",
                                "United Kingdom": "UK",
                                "Antigua and Barbuda": "Antigua",
                                "Congo, Dem. Rep.": "Democratic Republic of the Congo",
                                "Lao": "Laos",
                                "North Macedonia": "Macedonia",
                                "Cote d'Ivoire": "Ivory Coast",
                                "Trinidad and Tobago": "Trinidad",
                                "Micronesia, Fed. Sts.": "Micronesia",
                                "St. Vincent and the Grenadines": "Saint Vincent",
                                "St. Lucia": "Saint Lucia",
                                "St. Kitts and Nevis": "Saint Kitts",
                                "Slovak Republic": "Slovakia",
                                "Kyrgyz Republic": "Kyrgyzstan",
                                "Bolivia": "Boliva",
                                "Congo, Rep.": "Republic of Congo"},
                       inplace=True)

    # Check which countries still remain unmatched
    df_countries = set(df_to_match.columns.tolist())
    ol_regions = set(sorted(olympics.region.unique().tolist()))

    # Countries in the to_match df that are still not in the olympics df
    not_in_ol = df_countries.difference(ol_regions)
    # Countries in the olympics df that are still not in the to_match df
    not_in_df = ol_regions.difference(df_countries)

    # Printing not_in_ol and not_in_df shows which countries are still not matched. Used as a check.

    # save the resulting common countries
    common_countries = ol_regions.intersection(df_countries)

    return df_to_match, common_countries


def clean_data(df):
    """Transforms the olympics dataset so that it is easier to work with.

    Parameters
    ----------
    df : the olympics dataframe

    Returns
    ----------
    df : A dataframe of the olympics dataset after selecting the important data,
         removing unnecessary columns and filling missing values.
    """

    # Only select the entries from the summer season,
    # since the winter season is only for winter sports (ski, snowboard, etc.) and it might skew the results
    df = df.loc[df['Season'] == 'Summer']

    # Remove column "Games" because it is the same as the columns "Year" and "Season"
    if df['Games'].equals(df['Year'].map(str) + " " + df['Season']):
        # remove column "notes" as well
        df = df.drop(['Games', 'notes'], axis=1)

    # Drop null values from the region column, because they cannot be matched to a country
    # (which is what I am interested in)
    df = df[pd.notnull(df['region'])]

    # Replace the NaN values with "none"
    df['Medal'] = df['Medal'].fillna("none")

    # Missing values in the "Age", "Height" and "Weight" columns
    # Replace these missing values with the mean of the column
    # This allows the entries to be included in the visualization, without distorting the data (a lot)
    # After all, I am not interested in these variables, I am interested in whether the athletes won a medal
    df[['Age', 'Height', 'Weight']] = \
        df[['Age', 'Height', 'Weight']].fillna(value=df[['Age', 'Height', 'Weight']].mean())

    return df


def dummies_for_medals(df):
    """Creates dummy variables for the medals in the cleaned olympics dataset.

       This means that it will create one column for every unique value in the 'Medal' column.
       Then each entry gets the number 1 as value in the new column that corresponds to the value it has
       on the 'Medal' column. The columns that do not correspond get the number 0 as value.

       This makes it easier to calculate the sum of medals afterwards.

    Parameters
    -----------
    df : the olympics dataframe after having cleaned it.

    Returns
    -----------
    df : the olympics dataframe with dummy variables for the medals
    """

    # Create dummy variables for the medals in a new dataframe
    df_dummy_medals = pd.get_dummies(df['Medal'])

    # Concatenate the dummy dataframe with the main dataframe
    df = pd.concat([df, df_dummy_medals], axis=1)

    # Add a column with the total number of medals
    df['Total'] = df['Gold'] + df['Silver'] + df['Bronze']

    return df


def final_transformation(olympics, gdp, pop):
    """Creates one dataframe from the dataframes of olympics, gdp per capita, and population.

       The final dataframe contains the following measures for every country:
       - Gold
       - Bronze
       - Silver
       - N of medalists (raw)
       - mean_gdp
       - mean_pop
       - N of medalists (per 10m)

    Parameters
    -----------
    olympics : the olympics dataframe after having cleaned it and added dummies
    gdp      : the gdp per capita dataframe
    pop      : the population dataframe

    Returns
    -----------
    df       : a dataframe with all the necessary information from the given dataframes
    """

    # Group by country to get the total number of medalists
    olympics = olympics.groupby(['region'], as_index=False)[['Gold', 'Silver', 'Bronze', 'Total']].sum()
    olympics.rename(columns={'region': 'country',
                             'Total': 'N of medalists (raw)'}, inplace=True)

    # Get values for GDP per capita up to the last Olympics (measures start from 1960)
    gdp = gdp.loc[:'2016'].transpose()
    # Use the average GDP per capita over the years as a measure
    gdp['mean_gdp'] = gdp.mean(axis=1)
    gdp = gdp[['mean_gdp']].reset_index()

    # Get values for population only after the Olympics started and up to the last Olympics
    pop = pop['1896': '2016'].transpose().dropna()  # drop the country if there is no measure for population
    # Use the average of each country's population over the years as a measure
    pop['mean_pop'] = pop.mean(axis=1)
    pop = pop[['mean_pop']].reset_index()

    # THE FOLLOWING IS TO MAKE SURE THAT ONLY THE COUNTRIES FOR WHICH ALL DATA ARE AVAILABLE ARE INCLUDED
    # Dropping countries that are not in all dataframes
    olympics_countries = set(olympics.country.unique().tolist())
    gdp_countries = set(gdp.country.unique().tolist())
    pop_countries = set(pop.country.unique().tolist())

    to_drop = list(olympics_countries.difference(gdp_countries))
    to_drop2 = list(gdp_countries.difference(olympics_countries))
    to_drop.extend(to_drop2)
    to_drop3 = list(set(to_drop).difference(pop_countries))
    to_drop.extend(to_drop3)

    for country in to_drop:
        olympics = olympics.drop(olympics.index[olympics['country'] == country])

    # Merge the three datasets at once with the reduce function
    to_merge = [olympics, gdp, pop]
    df = reduce(lambda left, right: pd.merge(left, right, on='country'), to_merge)
    # -----------------------------------------------------------------------------------

    # Create a column with the number of medalists adjusted for population size
    # (medalists for every 10 million of a country's population)
    df['N of medalists (per 10m)'] = df['N of medalists (raw)'] * 10**7 / df['mean_pop']

    return df
