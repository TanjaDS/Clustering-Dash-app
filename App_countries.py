import numpy as np
from dash import Dash, html, dcc, Input, Output, dash_table, callback
import dash_bootstrap_components as dbc
import Data_countries as Data


# csv-files names and an index column
csv_names = ['missing_values', 'continent', 'annual-number-of-deaths-by-cause']
index = ['iso_code', 'iso_code', 'Code']

# define n_components of PCA, n_clusters of KMeans and AgglomerativeClustering
# a number of nearest neighbors and an effective minimum distance 
# between embedded points in UMAP
n, nKM, nAC, nUmap, dist_Umap = 0.9, 3, 3, 30, 0.1

# create an object, load and clean a data
df = Data.Visualization(csv_names, index, n, nKM, nAC, nUmap, dist_Umap)
df_cl, df_labels = Data.load_prepr(df)
# data preprocessing, train models
Data.train_model(df)


# create a list with columns' names
fig_names = df_cl.columns.tolist()

# reassign clusters' numbers in order to visualize them in a similar way
df_labels['ClusterKM'] = np.where(
    df_labels['ClusterKM'] == 1, 0, (np.where(
        df_labels['ClusterKM'] == 2, 2, 1)
        )
    )

# create a layout
layout = html.Div(
    children=[ 
        html.H1( # title of a layout
            children=
                'Cluster Analysis of Countries by Death Causes and '+\
                    'Socioeconomic Indicators',
        style={'text-align':'center'}),
        dcc.Graph( # call a PCA plot
            id='pca_', 
            figure=df.exp_pca(), 
            style={'display' : 'inline-block', 'width':'50%'}
            ),
        dcc.Graph( # call a 'silhouette score' plot
            id='silhouette_score_', 
            figure=df.silhouette(), 
            style={'display' : 'inline-block', 'width':'50%'}
            ),
        html.Br(), # make a space
        html.Br(), # make a space
        dcc.RadioItems( # create an option button
            id='clusters_',
            options = [
                {'label': 'KMeans', 'value': 'ClusterKM'},
                {'label': 'AgglomerativeClustering', 'value': 'ClusterAC'}
                ],
            value='ClusterKM',
            style={'text-align': 'center'},
            ),


    # add a container for a plot and a table
    html.Div([
        html.Div([

            html.Div(id='fig_plot_')

        ], style={'width': '65%', 'display': 'inline-block'}),

        # table container
        html.Div([
            dbc.Label('Difference between the Clustering Algorithms'),

            dash_table.DataTable(
                df.compair_cl().to_dict('records'), 
                [{"name": i, "id": i} for i in df.compair_cl().columns],
                style_data={ 'border': '1px solid blue' },
                style_header={'border': '2px solid pink',
                               'fontWeight': 'bold'},
                style_cell={'textAlign': 'left'},
                style_cell_conditional=[
                                     {
                                    'if': {'column_id': c},
                                    'textAlign': 'center'
                                     } for c in ['ClusterKM', 'ClusterAC']
                                 ],
                                 )

        ], style={
            'width': '30%', 
            'display': 'inline-block', 
            'marginTop' : '100px'},
            ),

    ], style={'display': 'flex'}),
    #https://stackoverflow.com/questions/69874454/plotly-dash-cannot-place-figure-and-table-side-by-side

    dcc.Dropdown( # cteate a dropdown menu
        id='fig_dropdown_',
        options=[{'label': x, 'value': x} for x in fig_names],
        value=fig_names[0],
        style={'text-align': 'center', 'width':'50%', 'display' : 'inline-block'}
        # use the first feature as the initial selection
    ),
    dcc.RadioItems( # create an option button
        id='graph_',
        options=[
            {'label': 'by Death Causes', 'value': 1},
            {'label': 'by Socioeconomic Indicators', 'value': 0}
            ],
        value=1,
        style={
            'text-align': 'center',  
            'marginLeft' : '100px', 
            'marginBottom' : '30px', 
            'display' : 'inline-block'},
            ),
    html.Br(), # make a space
    html.Div( # create a container for a plot
        id='fig_plot_1_', 
        style={
            'display' : 'inline-block', 'width':'50%'
            }
        ),
    html.Div( # create a container for a plot
        id='fig_plot_2_', 
        style={
            'display' : 'inline-block', 
            'width':'50%'
            }
        )
    ])

# define a callback for updating the plots
# based on the dropdown selection and the option buttons
@callback(
    [Output('fig_plot_', 'children'),
    Output('fig_plot_1_', 'children'),
    Output('fig_plot_2_', 'children')],
    [Input('fig_dropdown_', 'value'),
    Input('clusters_', 'value'),
    Input('graph_', 'value')]
    )

def update_figure(fig_name: str, cluster: str, graph_sel: int):
    """ call the plots
    Params: fig_name (str), cluster (str), graph_sel (int)
    Return: fig_treemap, fig_2, fig_m
    """ 
    fig_treemap = df.treemap(cluster)
    fig = df.umap_plot(fig_name, cluster)
    fig_m = df.cl_analys(graph_sel, cluster)
    

    return(dcc.Graph(figure=fig_treemap), 
            dcc.Graph(figure=fig), 
            dcc.Graph(figure=fig_m))

