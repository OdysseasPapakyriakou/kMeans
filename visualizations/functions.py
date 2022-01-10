# Name: Odysseas Papakyriakou
# Student_ID: 12632864

"""This module provides a number of functions to visualize data"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from transformed_datasets import *

from bokeh.plotting import figure
from bokeh.models import HoverTool, ColumnDataSource, Select
from bokeh.layouts import column, row

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score


def visualize_relationship(doc):
    """Creates a visualization of the relationship between countries' mean GDP and their number of medalists.
    It also groups the countries by using k means clustering from the sklearn library.

    The user can choose which metric to show for the number of medalists
    - grand total (raw)
    - adjusted for population size."""

    # The y axis, which is set by the user. It starts with he raw number of medalists
    y = "N of medalists (raw)"

    # The dataframe on which k means clustering is applied to estimate the groups
    X = final[['mean_gdp', y]]

    # Apply k means with 3 number of clusters
    # K means is not deterministic, which means that for every time it is run it may show a different grouping.
    # However, by default, the methods uses 300 iterations to calculate the clusters, so it results in the same
    # grouping every time
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X)
    y_kmeans = kmeans.fit_predict(X)

    # Add the groups to which the countries belong as a new column in the final dataframe
    final['map'] = y_kmeans.astype(str)

    # Map the groups to specific colors
    colormap = {'0': 'yellow', '1': 'purple', '2': 'blue'}
    colors = [colormap[x] for x in final['map']]
    final['colors'] = colors

    # Create the source
    source = ColumnDataSource(data={'x': final['mean_gdp'],
                                    'y': final[y],
                                    'Country': final['country'],
                                    'colors': final['colors']})

    cor_coef = np.corrcoef(final['mean_gdp'], final[y])[0, 1]

    # Create hover
    hover = HoverTool(
        tooltips=[
            ('Country', "@Country"),
            ('N of medalists', "@{y}{0.00}"),
            ('Mean GDP', "@{x}{0.00}")
        ])

    # Create the figure
    p = figure(plot_width=700,
               plot_height=400,
               title=f"Relationship between GDP/capita and Olympics performance. Correlation: {round(cor_coef, 2)}",
               x_axis_type="log",
               y_axis_type="linear",
               tools=[hover])

    p.title.align = "center"

    # Label the axes
    p.xaxis.axis_label = "GDP per capita (mean)"
    p.yaxis.axis_label = y

    # Create the glyph
    p.scatter(x='x', y='y', size=5, color='colors', source=source)

    # Create interaction using select so the user can select which metric to choose for the N of medalists
    # (raw or adjusted)
    select = Select(title="Select metric for medalists",
                    options=['N of medalists (raw)', 'N of medalists (per 10m)'],
                    value=y)

    # Callback function
    def update_metric(attrname, old, new):
        y = select.value
        source.data = {'x': final['mean_gdp'],
                       'y': final[y],
                       'Country': final['country'],
                       'colors': final['colors']}

        cor_coef = np.corrcoef(final['mean_gdp'], final[y])[0, 1]
        p.title.text = f"Relationship between GDP/capita and Olympics performance. Correlation: {round(cor_coef, 2)}"
        p.yaxis.axis_label = y

    # Update the plot on change by the user
    select.on_change('value', update_metric)

    # Create a layout
    layout = column(select, p)
    doc.add_root(layout)

    return layout


def kmeans_silhouette(column_a, column_b):
    """Calculates and plots the average silhouette for up to 6 clusters using k means clustering.

    The average silhouette is a metric that shows the performance of different numbers of clusters.
    It takes values from -1 to 1, and the closer the value is to 1, the better the performance.

    The code is largely based on the example on the sklearn library:
    https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html

    Parameters
    -----------
    column_a : a string representing the 'mean_gdp' column of the final dataframe
    column_b : a string representing either of the two columns of the final dataframe:
               - 'N of medalists (raw)' or the
               - 'N of medalists (per 10m)

    Plots the visualizations
    """

    # How many clusters to evaluate
    range_n_clusters = [2, 3, 4, 5, 6]

    # Create the data on which k means clustering and the average silhouette is applied
    X = final[[column_a, column_b]]

    # Make the plots
    for n_clusters in range_n_clusters:
        # 2 plot for each number of clusters
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # These limits allow all information to be clearly visible
        ax1.set_xlim([-0.1, 1])

        # Insert blank space between silhouette plots so they are clearly visible
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Run k means clustering
        clusterer = KMeans(n_clusters=n_clusters, init='k-means++', random_state=10)
        cluster_labels = clusterer.fit_predict(X)

        # Compute the silhouette score
        silhouette_avg = silhouette_score(X, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        # To be used to plot the center of the clusters
        y_lower = 10

        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        # Clear the y axis labels
        ax1.set_yticks([])
        # Set x axis labels
        ax1.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])

        # The 2nd subplot shows the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(X[column_a], X[column_b], marker='o', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

        # Label the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')

        ax2.set_title("The visualization of the clustered data")
        ax2.set_xlabel(f"{column_a}")
        ax2.set_ylabel(f"{column_b}")

        ax2.set_xscale('log')

        plt.suptitle(("Silhouette analysis for KMeans clustering on the data "
                      "with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')

    plt.show()


def kmeans_elbow_method(x, y):
    """Computes and plots the Within Cluster Sum of Squares (WCSS)
    for up to 6 clusters using k means clustering.

    The best number of clusters is the one after which the change in WCSS begins to level off.

    Parameters
    -----------
    x : a string representing the 'mean_gdp' columns of the final dataframe
    y : a string representing either of the two columns of the final dataframe:
        - 'N of medalists (raw)'
        - 'N of medalists (per 10m)'

    Plots the visualization
    """

    # The dataframe with the relevant columns
    X = final[[x, y]]
    # The list with the WCSS
    wcss = []
    # I use up to 6 clusters
    for i in range(1, 7):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, random_state=10)
        kmeans.fit(X)
        # WCSS is computed with inertia_
        wcss.append(kmeans.inertia_)

    # Make the plot
    plt.plot(range(1, 7), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Within Cluster Sum of Squares')

    plt.show()


def get_country_gdp(country):
    """This function is used in the 'visualize_performances' function to update the GDP per capita in the visualization.

    Parameters
    -----------
    country : a string representing the country for which to get the gdp per capita

    Returns
    -----------
    df : a dataframe with the measures for gdp per capita over the years for a given country
    """

    # Get the country's gdp
    df = gdp[country].to_frame()
    # Name the column
    df.columns = ['GDP']
    # Add a new column 'country' with the name of the country as value
    # This will be used when the user hovers over the values in the plot so it can show the country's name
    df['Country'] = country

    return df


def get_country_olympics(country):
    """This function is used in the 'visualize_performances' function to update the medalists in the visualization.

    It gathers all the medalists from a given country and it combines it with the country's population.
    The population is used to create a new column with the adjusted measure for the country's medalists
    (adjusted for population size, in this case: for every 10 million of citizens).

    Parameters
    -----------
    country : a string representing the country for which to get
              - the medalists from the olympics dataframe
              - the population from the population dataframe

    Returns
    ----------
    df : a dataframe with the total number of medalists and the number of medalists
         adjusted for population size, for a given country
    """

    # Get data only from the given country
    country_ol = olympics.loc[olympics['region'] == country]
    # Calculate the sum of medalists for every year
    country_ol = country_ol.groupby(['Year'], as_index=False)[['Gold', 'Silver', 'Bronze', 'Total']].sum()
    # Give the country's name as value to all entries in the region column
    # This will be used when the user hovers over the values in the plot so it can show the country's name
    country_ol['region'] = country

    # Get the population for the country
    p = pop[[country, 'Year']]

    # Get all the years for which there are measures for the medalists (in the country_ol dataframe) in a list
    olympic_years = [str(y) for y in country_ol['Year'].values.tolist()]   # str

    # Only get the population for these years
    p = p.loc[p['Year'].isin(olympic_years)]
    # Restructure dataframe so it is easier to use
    p = p.reset_index(drop=True).rename(columns={country: 'Population'})

    # Concatenate the two dataframes
    combined = pd.concat([country_ol, p], axis=1).iloc[:, 1:]
    # Create a new column with the number of medalists adjusted for population size
    combined['T per pop'] = (combined['Total'] * 10 ** 7) / combined['Population']

    return combined


def visualize_performances(doc):
    """Creates a visualization with 2 line-charts.

    The first line-chart shows the performance of 2 different countries on the Olympics across the years.
    For each country the plot shows the total number of medalists (continuous line), as well as
    the number of medalists adjusted for population size (dashed line).

    The second line-chart shows how the GDP per capita of these 2 countries changes over the years.

    The user can choose which countries to visualize.
    """

    # Visualize Jamaica and Japan by default
    a = "Jamaica"
    b = "USA"

    # The sources for the first plot change by user input, so the function 'get_country_olympics' is used
    source1 = ColumnDataSource(data=get_country_olympics(a))
    source2 = ColumnDataSource(data=get_country_olympics(b))

    # The sources for the second plot change by user input, so the function 'get_country_gdp' is used
    source3 = ColumnDataSource(data=get_country_gdp(a))
    source4 = ColumnDataSource(data=get_country_gdp(b))

    # Create hover for the first plot
    hover_ol = HoverTool(
        tooltips=[
            ('Year', "@Year"),
            ('Total', "@Total"),
            ('Adjusted', "@{T per pop}{0.00}"),
            ('Golds', "@Gold"),
            ('Silvers', "@Silver"),
            ('Bronzes', "@Bronze"),
            ('Country', "@region")
        ]
    )

    # Create hover for the second plot
    hover_gdp = HoverTool(
        tooltips=[
            ('Year', "@Year"),
            ('GDP', "@GDP"),
            ('Country', "@Country")
        ]
    )

    # Create figure for the first plot
    ol = figure(plot_height=400,
                plot_width=400,
                tools=[hover_ol],
                title=f"Comparing medals for {a} and {b}")

    # Create figure for the second plot
    g = figure(plot_height=400,
               plot_width=400,
               tools=[hover_gdp],
               title=f"Comparing GDP for {a} and {b}")

    # Title
    ol.title.align = "center"
    g.title.align = "center"

    # x axes
    ol.xaxis.axis_label = "Years"
    ol.xaxis.axis_label_text_font_style = "bold"

    g.xaxis.axis_label = "Years"
    g.xaxis.axis_label_text_font_style = "bold"

    # y axes
    ol.yaxis.axis_label = "Number of medalists"
    ol.yaxis.axis_label_text_font_style = "bold"

    g.yaxis.axis_label = "GDP per capita in US dollars (adjusted)"    # divided by midyear population
    g.yaxis.axis_label_text_font_style = "bold"

    # GLYPHS

    # Continuous line of 1st plot: total number of medalists over the years
    ol.line('Year', 'Total', line_color='lightcoral', line_width=2,
            alpha=0.7, legend_label="Country 1: Total", source=source1)
    ol.line('Year', 'Total', line_color='black', line_width=2,
            alpha=0.7, legend_label="Country 2: Total", source=source2)

    # Dashed line of first plot: medalists adjusted for population size over the years
    ol.line('Year', 'T per pop', line_dash="4 4", line_color='lightcoral', line_width=2,
            alpha=0.7, legend_label="Country 1: Adjusted", source=source1)
    ol.line('Year', 'T per pop', line_dash="4 4", line_color='black', line_width=2,
            alpha=0.7, legend_label="Country 2: Adjusted", source=source2)

    # Second plot: GDP over the years
    g.line('Year', 'GDP', line_color='lightcoral', line_width=2, alpha=0.7, legend_label="Country 1", source=source3)
    g.line('Year', 'GDP', line_color='black', line_width=2, alpha=0.7, legend_label="Country 2", source=source4)

    # Legend location
    ol.legend.location = "top_left"
    g.legend.location = "top_left"

    # Create interaction using select so the user can select which 2 countries to visualize
    select_c1 = Select(title="Compare country 1:", value=a, options=countries, width=300)
    select_c2 = Select(title="to country 2:", value=b, options=countries, width=300)

    # CALLBACK FUNCTIONS
    def update_c1(attrname, old, new):
        a = select_c1.value
        b = select_c2.value
        source1.data = get_country_olympics(a)
        source3.data = get_country_gdp(a)

        ol.title.text = f"Comparing medalists for {a} and {b}"
        g.title.text = f"Comparing GDP for {a} and {b}"

    # Update the plot on change by the user
    select_c1.on_change('value', update_c1)

    def update_c2(attrname, old, new):
        a = select_c1.value
        b = select_c2.value
        source2.data = get_country_olympics(b)
        source4.data = get_country_gdp(b)

        ol.title.text = f"Comparing medalists for {a} and {b}"
        g.title.text = f"Comparing GDP for {a} and {b}"

    # Update the plot on change by the user
    select_c2.on_change('value', update_c2)

    # Create a layout
    layout = column(row(select_c1, select_c2), row(ol, g))
    doc.add_root(layout)

    return layout
