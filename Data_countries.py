import pandas as pd
import numpy as np
import wbgapi #pip install wbgapi, data from www.worldbank.org
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score
import umap
import plotly.io as pio
import plotly.express as px
pio.templates.default = 'plotly_dark' # default plotly theme
import warnings
warnings.filterwarnings("ignore")


class Model:
    """ read csv files, impute, fit and transorm a data
        make dimension reduction with PCA and UMAP, predict clusters
        - csv_names (list) - list with csv-files
        - index (str) - set a column as index
        - n (float) - assign selected explained variance in PCA 
                            to choose a number of components
        - nKM (int) - assign a number of clusters in KMeans
        - nAC (int) - assign a number of clusters in AgglomerativeClustering
        - nUMAP (int) - assign a number of nearest neighbors in UMAP
        - dist_UMAP (float) - assign an effective minimum distance 
                                between embedded points in UMAP
    """ 
    def __init__(self, csv_names: list, index: str, 
                    n: float, nKM: int, nAC: int, 
                    nUmap: int, dist_Umap: float):
        """ initialize Model
        Params: csv_names (list), index (str), 
                    n (float), nKM (int), nAC (int), 
                    nUmap (int), dist_Umap (float)
        """
        # https://www.geeksforgeeks.org/read-multiple-csv-files-into-separate-dataframes-in-python/
        self.df = [] 
        for i in range(len(csv_names)):
            load_df = pd.read_csv(csv_names[i] + '.csv').set_index(index[i])
            self.df.append(load_df) #store datasets in a list
        self.n, self.nKM, self.nAC = n, nKM, nAC
        self.nUmap, self.dist_Umap = nUmap, dist_Umap

    def pipeline(self, pca=None):
        """ create a pipeline, 
            impute NAs, scale, fit and transform a data,
            assign number of components in PCA to 'n'
            for clustering and UMAP, to None for PCA visualization
        Args: pca (str, default=None)
        Return: fitted and transformed dataset
        """
        if pca == 'viz':
            n_comp = None
        else:
            n_comp = self.n

        self.pipe = Pipeline([
                        ('imputer', KNNImputer(missing_values=np.nan)),
                        ('scaler', StandardScaler()),
                        ('reducer', PCA(n_comp))
                    ])
        self.X = self.pipe.fit_transform(self.df_cl)
        return self.X

    def clustering(self, algorithm: str, cl_name: str):
        """ predict clusters, assign clusters to a dataframe
        Args: algorithm (str), cl_name (str)
        Return: None
        """
        algorithm.fit_predict(self.pipeline())
        self.df_labels[cl_name] = algorithm.labels_
        

    def umap_reducer(self, reducer: str):
        """ arrange a data in low-dimensional space using UMAP algorithm,
            assign 'x' and 'y' to a dataframe
        Args: reducer (str)
        Return: None
        """
        self.df_labels['x'] = reducer.fit_transform(self.X)[:,0]
        self.df_labels['y'] = reducer.fit_transform(self.X)[:,1]

class Visualization(Model):
    """ the child class "Visualization" that inherits all from the parent class "Model"
    """ 
    def exp_pca(self):
        """ compute the principal components (all features),
            explained varince and comulative expl. variance
            plot all the components and cumulative expl. variance,
            color a number of components which will be using in clustering further
        Args: None
        Return: fig (px.bar)
        """
        # fit and transform all features
        pca = PCA()
        pca.fit_transform(self.pipeline('viz'))
        # return explained variance ratio
        exp_pca = pca.explained_variance_ratio_ * 100
        x = range(1,len(exp_pca)+1)
        # plot a bar chart
        fig = px.bar(
            np.cumsum(exp_pca), x=x, y=0,
            # add annotations over each component 
            # with an expl. variance of each component
            text=exp_pca.round(2), 
            title = 'Principal Component Analysis'
            )
        # color selected components based on selected cum. expl. variance
        # update hoverdata
        fig.update_traces(
            marker_color=[
                'green' if x < self.n*100 else 'blue' for x in np.cumsum(exp_pca)
                ], 
            marker_line_color='rgb(8,48,107)',
            marker_line_width=1.5,
            hovertemplate='<br>Component %{x}'+\
                           '</br>Explained Variance %{text}'+\
                               '</br>Cumulative Explained Variance %{y:,.2f}'
        )

        fig.update_xaxes(title_text='Number of Components')
        fig.update_yaxes(title_text='Cumulative Explained Variance')

        return fig

    def silhouette(self):
        """ compute average silhouette scores for each cluster,
            and numbers of obsevations in each cluster,
            transform a list and dataframe to a new dataframe to plot the data,
            plot the data
        Args: None
        Return: fig (px.bar)
        """
        s_avg_list, count_clusters = [], pd.DataFrame({})
        # compute av. sil. score for 2,3,4,5,6 clusters
        for n_clusters in range(2,7):
            opt_km = KMeans(n_clusters=n_clusters, random_state=44)
            opt_cluster_labels = opt_km.fit_predict(self.pipeline())
            silhouette_avg = silhouette_score(self.X, opt_cluster_labels)
            count_clusters[n_clusters] = opt_cluster_labels
            s_avg_list.append([n_clusters, silhouette_avg])
        # inspiration - https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
        # https://stackoverflow.com/questions/57242200/how-to-do-value-counts-on-each-row-and-make-some-columns-whose-values-are-the-co
        # sum numbers of observations in each cluster
        t_count_clusters = count_clusters.T.apply(pd.Series.value_counts, axis=1)
        t_count_clusters.fillna(0, inplace=True)
        # compute av. sil. score in same ratio as the number of observations
        # Note! this is only for visualization!
        t_count_clusters = t_count_clusters.\
                            div(len(self.df_cl)).\
                            mul(np.array(s_avg_list)[:,1], axis=0)
        fig = px.bar(t_count_clusters, 
                    x=[5,4,3,2,1,0], 
                    orientation='h',
                    color_discrete_sequence=[
                        '#46039f','#fdca26',
                        '#d8576b','#f0f921',
                        '#bd3786','#0d0887'
                        ],
                    text=[
                        f'<br>{i}' for i in t_count_clusters.sum(axis=1).round(2)
                        ],
                    title = 'Silhouette Score and Proportion '+\
                        'of Observations per Cluster (KMeans)'
                    )
        fig.update_layout({'bargap' : 0.7}, showlegend=False)
        # remove text from each cluster and place only in one cluster
        # https://getridbug.com/python/hide-select-stacked-bar-text-annotation-in-plotly-express/
        fig.for_each_trace(
            lambda t: t.update(text = []) if t.name not in ['0'] else ()
            )
        fig.update_traces(textfont_size=12, 
                        textangle=0, 
                        textposition="outside",
                        hovertemplate='<br>')
        fig.update_xaxes(title_text='Average Silhouette Score', 
                            range=[0,0.35])
        fig.update_yaxes(title_text='Number of Clusters')
        return fig



    def treemap(self, cl_name: str):
        """ plot a treemap
        Args: cl_name (str)
        Return: fig (px.treemap)
        """

        fig = px.treemap(
            self.df_labels, 
            # create a container with continents and countries
            path=[px.Constant('world'), 'Continent', 'Country'], 
            values='Life expectancy', # defalt value
            color=cl_name,
            hover_name='Country',
            title = 'Clustering by Country',
            template='plotly'
            )
        fig.update(layout_coloraxis_showscale=False)
        #https://community.plotly.com/t/change-hovertext-fields-of-a-treemap-in-plotly-express/42520/2
        fig.update_traces(
            hovertemplate='%{label}<br>Life expectancy: %{value:.2f}\
                                <br>Cluster: %{color}'
            )
        return fig

    def cl_analys(self, graph_sel: int, cl_name: str):
        """ subset the dataframe with assigned clusters
            grouping by average death causes
            and socio-economic indicators per each cluster;
            transform the new dataframe to a long-form  dataframe;
            plot the data, selecting the 30 largest values, 
            sorted by descending order
        Args: graph_sel (int), cl_name (str)
        Return: fig_m (px.histogram)
        """
        start, end = [0, 9], [9, 40]
        # group by death causes and indicators and transform the dataframe to a longformat table
        self.means = self.df_labels.groupby(cl_name)\
            [self.df_labels.columns[
                start[graph_sel]:end[graph_sel]]
                ].mean().astype(int).reset_index()
        self.means_m = pd.melt(self.means, id_vars=cl_name)
        
        # plot, select by a cluster and by a group of indicators from dropdown
        fig_m = px.histogram(
                self.means_m.sort_values('value', 
                            ascending=False).nlargest(30, 'value'), 
                y='variable', x='value', 
                # color and separate by a cluster
                color=cl_name, 
                facet_col=cl_name, 
                color_discrete_map={
                                1: '#bd3786',
                                2: '#f0f921',
                                0: '#0d0887'
                            },
                title='Average per Cluster',
                template='plotly_dark'
                )
    #https://stackoverflow.com/questions/59057881/python-plotly-how-to-customize-hover-template-on-with-what-information-to-show
        fig_m.update_traces(
                hovertemplate='<br>' + 'Average %{y}: %{x}'
                )

        fig_m.update_layout(showlegend=False)
        fig_m.update_yaxes(title_text=None)
        fig_m.update_xaxes(title_text=None)
        return fig_m


    def umap_plot(self, col: str, cl_name: str):
        """ plot UMAP results in 3-dimensions
        Args: col (str), cl_name ( str)
        Return: fig (px.scatter_3d)
        """
        fig = px.scatter_3d(
            self.df_labels, 
            x='x', y='y', z=self.df_labels[col],
            hover_name='Country',
            hover_data = {
                        'Continent':True, 
                        'x':False, 
                        'y':False
                    },
            color=cl_name, 
            title='Uniform Manifold Approximation and '+\
                     'Projection for Dimension Reduction',
            template='plotly_dark'
            )

        fig.update_traces(
            marker=dict(
                size=6,
                line=dict(
                    width=0.5,
                    color='lightgrey'
                    )
                ),
            selector=dict(
                mode='markers'
                )
            )
        fig.update(layout_coloraxis_showscale=False)
        return fig

    def compair_cl(self):
        """ compute a new column with difference 
            between the two cluster algorithms
        Args: None
        Return: df_subset (pandas.DataFrame)
        """
        self.df_labels['diff'] = abs(self.df_labels['ClusterKM'] -\
                                         self.df_labels['ClusterAC'])
        df_subset = self.df_labels[self.df_labels['diff'] != 0]
        
        return df_subset[['Country', 'ClusterKM','ClusterAC']]

def get_wb_data():
    """ get a data from www.worldbank.org
    Params: None
    Return: df_world_bank (pandas.DataFrame)
    """ 
    # https://pypi.org/project/wbgapi/
    indicators = ['SH.XPD.CHEX.GD.ZS', 'SH.H2O.BASW.ZS',
                'SP.POP.TOTL','SP.DYN.LE00.FE.IN','SP.POP.65UP.TO.ZS',
                'SP.POP.1564.TO.ZS','SP.POP.0014.TO.ZS']
    df_world_bank = wbgapi.data.DataFrame(indicators, time=2019)
    df_world_bank.loc['HRV', 'SH.H2O.BASW.ZS'] = wbgapi.data.get(
                            'SH.H2O.BASW.ZS', 'HRV', time=2007)['value']
    missing_country = [
        ['ALB', 2018], ['YEM', 2015], ['SYR', 2012], ['LBY', 2011]
        ]
    for i in range(len(missing_country)):
        country, year = missing_country[i][0], missing_country[i][1]
        ind = 'SH.XPD.CHEX.GD.ZS'
        df_world_bank.loc[country, ind] = wbgapi.data.get(
                                    ind, country, time=year)['value']
    missing_values = ['ATG','ARG','HRV','DMA', 'GNQ', 'GRD','KNA','VCT']
    for i in missing_values:
        df_world_bank.loc[i, 'SH.H2O.BASW.ZS'] = wbgapi.data.get(
                                'SH.H2O.BASW.ZS', i, time=2016)['value']
    new_name_col = [
                'Drinking water services (% of population)', 
                'Current health expenditure (% of GDP)', 
                'Life expectancy', 
                'Population ages 0-14', 
                'Population ages 15-64',
                'Population ages 65 and above', 
                'population'
            ]
    df_world_bank.columns = new_name_col
    return df_world_bank

def load_prepr(self, df_world_bank=get_wb_data()):
    """ get a data from csv-files, 
        merge all dataframes,
        clean a data,
        copy a dataframe,
        returns two dataframes
    Params: df_world_bank (pandas.DataFrame)
    Return: df_cl (pandas.DataFrame), df_labels (pandas.DataFrame)
    """ 
    df_world_bank = self.df[1].merge(
        df_world_bank, how='left', left_index=True, right_index=True
        )
#https://stackoverflow.com/questions/29177498/python-pandas-replace-nan-in-one-column-with-value-from-corresponding-row-of-sec
    df_world_bank = df_world_bank.fillna(self.df[0])
    # select only 2019
    self.df[2] = self.df[2][self.df[2]['Year'] == 2019]
    self.df[2].drop([
                'Year', 'Entity', 
                'Number of executions (Amnesty International)', 
                'Terrorism (deaths)'
            ], axis=1, inplace=True)
    self.df[2].columns = self.df[2].columns.\
                            str.replace('Deaths - ', '', regex=False)
    self.df[2].columns = self.df[2].columns.\
                            str.replace(
                                ' - Sex: Both - Age: All Ages (Number)', '', 
                                    regex=False)
    self.df_cl = self.df[2].merge(
        df_world_bank['population'], 
        how='inner', 
        left_index=True, right_index=True
        )
    # find a ratio of deaths on 100000 people
    self.df_cl = (self.df_cl.\
                    div(self.df_cl['population'], axis=0) * 100000).\
                        round(3)
    self.df_cl = df_world_bank.merge(
                    self.df_cl, 
                    how='inner', 
                    left_index=True, right_index=True
                    )
    self.df_cl.drop(['population_x', 'population_y'], axis=1, inplace=True)
    col = ['Current health expenditure (% of GDP)', 'Life expectancy',
            'Population ages 0-14', 'Population ages 15-64',
        'Population ages 65 and above']
    # fillna in Greenland with the data from Danmark
    self.df_cl.loc['GRL', col] = self.df_cl.loc['DNK', col]
    # drop Eritrea and Tokelau (no deaths data), and World 
    self.df_cl.drop(['ERI', 'OWID_WRL', 'TKL'], axis=0, inplace=True)
    # copy the dataframe, a new dataframe will be used to assign clusters
    self.df_labels = self.df_cl.copy()
    self.df_cl.drop(['Country', 'Continent'], axis=1, inplace=True)
    return self.df_cl, self.df_labels

def train_model(self):
    """ call a clustering method,
        call an umap_reducer method
    Params: None
    Return: None
    """ 
    self.clustering(KMeans(n_clusters=self.nKM, 
                    random_state=44), 
                    cl_name='ClusterKM')
    self.clustering(AgglomerativeClustering(
                    n_clusters=self.nAC), 
                    cl_name='ClusterAC')
    self.umap_reducer(umap.UMAP(n_neighbors=self.nUmap, 
                    min_dist=self.dist_Umap, 
                    random_state=1))

