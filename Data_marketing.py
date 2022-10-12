import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score
import umap
import hdbscan
import plotly.io as pio
import plotly.express as px
pio.templates.default = 'seaborn'
import warnings
warnings.filterwarnings("ignore")


class Model:
    """ read csv files, impute, fit and transorm a data
        make dimension reduction with PCA and UMAP, predict clusters
        - path (list) - list with csv-files
        - date_col (str) - assign a date column to parse
        - n (int) - assign a number of components PCA
        - nKM (int) - assign a number of clusters in KMeans
        - nAC (int) - assign a number of clusters in AgglomerativeClustering
        - nUMAP (int) - assign a number of nearest neighbors in UMAP
        - dist_UMAP (float) - assign an effective minimum distance 
                                between embedded points in UMAP
    """ 
    def __init__(self, path: str, date_col: str, 
                        n: int, nKM: int, nHDB: int, nUmap: int, dist_Umap: float):
        """ initialize Model
        Params: path (str), date_col (str), 
                    n (int), nKM (int), nAC (int), 
                    nUmap (int), dist_Umap (float)
        """
        self.df = pd.read_csv(path, sep='\t', 
                            parse_dates=[date_col], 
                            infer_datetime_format=True)
        self.n, self.nKM, self.nHDB = n, nKM, nHDB
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
                    ('imputer', SimpleImputer(
                                    strategy = 'median', 
                                    missing_values=np.nan)),
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
                    # add annotations over each component with an 
                    # expl. variance of each component
                    text=exp_pca.round(2), 
                    title = 'Principal Component Analysis'
                )
        # color selected components based on selected cum. expl. variance
        # update hoverdata
        fig.update_traces(
            marker_color=[
                'green' if x<np.cumsum(exp_pca)[self.n]\
                     else 'blue' for x in np.cumsum(exp_pca)
                ], 
            marker_line_color='rgb(8,48,107)',
            marker_line_width=1.5, 
            opacity=0.5,
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
        t_count_clusters = t_count_clusters.div(len(self.df_cl)).\
                                mul(np.array(s_avg_list)[:,1], axis=0)
        fig = px.bar(
                t_count_clusters, x=[5,4,3,2,1,0], orientation='h',
                color_discrete_sequence=px.colors.qualitative.Set2_r[2:],
                opacity=0.7,
                text=[
                    f'{i}' for i in t_count_clusters.sum(axis=1).round(2)
                    ],
                title = 'Silhouette Score and Proportion '+\
                            'of Observations per Cluster (KMeans)'
            )
        # remove text from each cluster and place only in one cluster
        #https://getridbug.com/python/hide-select-stacked-bar-text-annotation-in-plotly-express/
        fig.for_each_trace(
            lambda t: t.update(text = []) if t.name not in ['0'] else ()
            )
        fig.update_traces(textfont_size=12, 
                        textangle=0, 
                        textposition="outside",
                        hovertemplate='<br>')
        fig.update_layout({'bargap' : 0.7}, showlegend=False)
       
        fig.update_xaxes(title_text='Average Silhouette Score', range = [0,0.55])
        fig.update_yaxes(title_text='Number of Clusters')
        return fig


    def cl_analys(self, col: str, cl_name: str, main_colors: list):
        """ subset the dataframe with assigned clusters
            grouping by product category, marketing action,
            average income and age per each cluster;
            transform the new dataframe to a long-form  dataframe;
            plot the data, sorted by descending order
        Args: col (str), cl_name (str), main_colors (list)
        Return: fig_cat (px.bar_polar), fig_inc (px.bar),
                fig_mar (px.funnel)
        """
        self.means = self.df_labels.groupby(cl_name)[col].mean().round(1).reset_index()
        self.means_m = pd.melt(self.means, id_vars=cl_name)
        # remove Cluster -1 (outliers) if a selected algoritm is HDBSCAN
        if cl_name == 'ClusterHDB':
            self.means_m = self.means_m[self.means_m[cl_name] != -1]
            main_colors = ['rgb(128,177,211)','rgb(141,211,199)']
        # select omly marketing actions
        col_sub = [
            'NumPurchasesTotal', 'NumWebVisitsMonth', 
            'NumDealsPurchases', 'NumDealsResponse'
            ]
        self.df_subs = self.means_m[self.means_m['variable'].\
                    isin(col_sub)].sort_values('value', ascending=False)
        # plot spending ratio by product category per each cluster
        fig_cat = px.bar_polar(
                    self.means_m[self.means_m['variable'].isin(col[11:])], 
                    r='value', theta='variable', 
                    color=cl_name, 
                    color_continuous_scale=main_colors,
                    title='Average % of Each Category '+\
                            'in Total Spendnig per Cluster'
                )
        fig_cat.update(layout_coloraxis_showscale=False)
        # plot average income and average age per each cluster
        fig_inc = px.bar(self.means_m[self.means_m['variable']==col[0]], 
                            x=cl_name, y='value', 
                            color=cl_name,
                            color_continuous_scale=main_colors,
                            title='Average Income and Age per Cluster')
        fig_inc.update_traces(
            hovertemplate='Income: %{y}',
            text = [
                f'Average Age: {int(x)}' for x\
                     in self.means_m[self.means_m['variable']=='Age']['value']
                     ]
                )
        fig_inc.update(layout_coloraxis_showscale=False)
        fig_inc.update_yaxes(title_text='Average Income')
        # plot marketing actions 
        stages = col_sub
        value_cl = self.df_subs[cl_name].unique().tolist()
        value_cl.sort()
        df_0 = pd.DataFrame(
            dict(
                value=self.df_subs[self.df_subs[cl_name] == 0]['value'], 
                stage=stages, 
                Cluster=0))
        for i in value_cl[1:]:
            df_x = pd.DataFrame(
                dict(
                    value=self.df_subs[self.df_subs[cl_name] == i]['value'], 
                    stage=stages, 
                    Cluster=i))
            df = pd.concat([df_0, df_x], axis=0)
            df_0 = df

        fig_mar = px.funnel(
                    df, 
                    x='value', y='stage', 
                    color='Cluster', 
                    color_discrete_sequence=main_colors,
                    title='Average Action per Cluster'
                )

        fig_mar.update_yaxes(title_text=None)

        return fig_cat, fig_inc, fig_mar

    def umap_plot(self, col: str, cl_name: str, main_colors: list):
        """ plot UMAP results in 3-dimensions
        Args: col (str), cl_name ( str), main_colors (list)
        Return: fig (px.scatter_3d)
        """
        # select colors for HDBSCAN
        if cl_name == 'ClusterHDB':
            main_colors = [
                'rgb(251,128,114)',
                'rgb(128,177,211)',
                'rgb(141,211,199)'
                ]
        fig = px.scatter_3d(
                self.df_labels, 
                x='x', y='y', z=self.df_labels[col],
                color=cl_name, 
                color_continuous_scale=main_colors,
                hover_data = {
                        'x':False, 
                        'y':False
                    },
                title='Uniform Manifold Approximation '+\
                    'and Projection for Dimension Reduction'
                )
        fig.update_traces(
            marker=dict(
                size=6,
                line=dict(
                    width=0.1,
                    color='white'
                    )
                ),
                selector=dict(
                    mode='markers'
                    )
                )
        fig.update(layout_coloraxis_showscale=False)
        return fig

def load_prepr(self):
    """ get a data from a csv-file, 
        clean a data,
        copy a dataframe,
        returns two dataframes
    Params: None
    Return: df_cl (pandas.DataFrame), df_labels (pandas.DataFrame)
    """ 
    # drop an observation with income 666666 (outlier),
    # drop the observations with age over 100 years (outliers)
    self.df.drop([2233, 192, 239, 339], axis=0, inplace=True)
    # find age from year of birth
    self.df['Age'] = 2014 - self.df['Year_Birth']
    # transform the data
    self.df['Marital_Status'] = np.where(
        self.df['Marital_Status'].isin(['Married', 'Together']), 2, 1)
    self.df['Kids at home'] = np.where(
        (self.df['Kidhome'] + self.df['Teenhome']) > 0, 1, 0)
    self.df['Family size'] = np.where(
        self.df['Marital_Status'] == 'Together', 2, 1) +\
             self.df['Kidhome'] + self.df['Teenhome']
    self.df['NumPurchasesTotal'] = self.df[[
        'NumWebPurchases',
        'NumCatalogPurchases',
        'NumStorePurchases'
            ]].sum(axis=1)
    self.df['DaysAsCustomer'] = (pd.to_datetime('2014-12-31') -\
                                     self.df['Dt_Customer']).dt.days
    self.df.columns = self.df.columns.\
                        str.replace(('Products|Mnt|Prods'), '', regex=True)
    # compute a proportion of each category in total spending
    col = ['Wines','Fruits','Meat','Fish','Sweet','Gold']
    self.df[col] = (self.df[col].\
                        div(self.df[col].\
                            sum(axis=1), axis=0) * 100).astype(int)
    col_deals = [
        'AcceptedCmp3',
        'AcceptedCmp4',
        'AcceptedCmp5',
        'AcceptedCmp1',
        'AcceptedCmp2',
        'Response'
        ]
    # compute total Deal Response
    self.df['NumDealsResponse'] = self.df[col_deals].sum(axis=1)
    self.df.drop(col_deals, axis=1, inplace=True)
    col_to_keep = [
                'Income', 'Age', 'Marital_Status', 'Family size', 
                'Kids at home', 'DaysAsCustomer', 
                'Recency', 'NumPurchasesTotal',
                'NumDealsResponse', 'NumDealsPurchases', 
                'NumWebVisitsMonth',
                'Wines', 'Fruits', 'Meat',
                'Fish', 'Sweet', 'Gold'
            ]
    self.df_cl = self.df[col_to_keep]
    # copy the dataframe, a new dataframe will be used to assign clusters
    self.df_labels = self.df_cl.copy()
    return self.df_cl, self.df_labels

def train_model(self):
    """ call a clustering method,
        call an umap_reducer method
    Params: None
    Return: None
    """ 
    self.clustering(KMeans(
                    n_clusters=self.nKM, 
                    random_state=44), 
                    cl_name='ClusterKM')
    self.clustering(hdbscan.HDBSCAN(
                    min_cluster_size=self.nHDB), 
                    cl_name='ClusterHDB')
    self.umap_reducer(umap.UMAP(
                    n_neighbors=self.nUmap, 
                    min_dist=self.dist_Umap, 
                    random_state=1))

                   