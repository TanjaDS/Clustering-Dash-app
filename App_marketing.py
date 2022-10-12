from dash import Dash, dcc, html, Input, Output, callback
import Data_marketing as Data


# define n_components of PCA, n_clusters of KMeans and AgglomerativeClustering
# a number of nearest neighbors and an effective minimum distance 
# between embedded points in UMAP
n, nKM, nHDB, nUmap, dist_Umap = 2, 3, 17, 30, 0.1 

# create an object, load and clean a data
df = Data.Visualization(
    'marketing_campaign.csv', 
    'Dt_Customer', 
    n, nKM, nHDB, nUmap, dist_Umap
    )
df_cl, df_labels = Data.load_prepr(df)
# data preprocessing, train models
Data.train_model(df)

# create a list with columns' names 
# create a list with main colors
fig_names = df_cl.columns.tolist()
main_colors = ['rgb(141,211,199)', 'rgb(251,128,114)', 'rgb(128,177,211)']

layout = html.Div(
    children=[ # create a container
        html.H1( # layout's title
            children='Cluster Analysis of Marketing Data',
            style={'text-align':'center'}
            ),
        dcc.Graph( # call a 'PCA' plot
            id='pca', 
            figure=df.exp_pca(), 
            style={'display' : 'inline-block', 'width':'50%'}
            ),
        dcc.Graph( # call a 'silhouette score' plot
            id='silhouette_score', 
            figure=df.silhouette(), 
            style={'display' : 'inline-block', 'width':'50%'}
            ),
        dcc.RadioItems( # create an option button
            id='clusters',
            options = [
                {'label': 'KMeans', 'value': 'ClusterKM'},
                {'label': 'HDBSCAN', 'value': 'ClusterHDB'}
                ],
            value='ClusterKM',
            style={'text-align': 'center'}
            ),
        html.Br(), # make a space
        dcc.Dropdown( # create a dropdown menu
            id='fig_dropdown',
            options=[{'label': x, 'value': x} for x in fig_names],
            value=fig_names[0], 
            style={
                'width':'65%', 
                'text-align': 'center', 
                'marginLeft' : '20px'
                }  # use the first borough as the initial selection
        ),
    # add a container for the figures
    html.Div( 
        id='fig_plot', 
        style={'display' : 'inline-block', 'width':'50%'}
        ),
    html.Div(
        id='fig_plot_1', 
        style={'display' : 'inline-block', 'width':'50%'}
        ),
    html.Div(
        id='fig_plot_2', 
        style={'display' : 'inline-block', 'width':'50%'}
        ),
    html.Div(
        id='fig_plot_3', 
        style={'display' : 'inline-block', 'width':'50%'}
        )
    ])

# define a callback for updating the figures
# based on the dropdown selection
@callback([
    Output('fig_plot', 'children'),
    Output('fig_plot_1', 'children'),
    Output('fig_plot_2', 'children'),
    Output('fig_plot_3', 'children')
    ],
    [
    Input('fig_dropdown', 'value'),
    Input('clusters', 'value')
    ])
def update_figure(fig_name: str, cluster: str):
    """ call the plots
    Params: fig_name (str), cluster (str)
    Return: fig_cat, fig_inc, fig_mar, fig
    """ 
    fig_cat, fig_inc, fig_mar = df.cl_analys(fig_names, cluster, 
                                        main_colors=main_colors)
    fig = df.umap_plot(fig_name, cluster,  main_colors=main_colors)

    return(
        dcc.Graph(figure=fig),
        dcc.Graph(figure=fig_inc),
        dcc.Graph(figure=fig_cat),
        dcc.Graph(figure=fig_mar)
        )

