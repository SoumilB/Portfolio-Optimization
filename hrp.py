import pandas as pd
import scipy.cluster.hierarchy
import scipy.spatial.distance
import numpy as np
import matplotlib.pyplot as plt
import collections
import random

class HRP:
    """
    Calculates the weights of the portfolio for all methodologies. 
    """
    def __init__(self, prior_data=None, file_path=None, start_date=None, end_date=None,):
        if prior_data is not None:
            self.returns =  self._generate_returns(data=prior_data, start_date=start_date, end_date=end_date)
        else:
            self.returns =  self._generate_returns(file_path=file_path, start_date=start_date, end_date=end_date)
            
        self.corr,self.cov = self.returns.corr(), self.returns.cov()
        self.clusters = None
        self.weights = None
        self.d = None
    
    def _generate_returns(self, data=None, file_path=None, start_date=None, end_date=None):
        if data is not None:
            stock_df = data
        else:
            stock_df = pd.read_csv(file_path)
        
        stock_df['Date'] = pd.to_datetime(stock_df['Date'])

        stock_datedf = stock_df
        
        if start_date and end_date:
            stock_datedf = stock_df[(stock_df['Date'] >= start_date) & (stock_df['Date'] <= end_date)]
        
        stock_datedf.set_index('Date', inplace=True)
        returns_df = stock_datedf.pct_change().dropna(how="all").dropna(axis=1, how="all") ## Drop rows and column with all NA 
        
        return returns_df.fillna(0) ## Fill Na values with 0 returns
    
    def _get_clusters(self,returns,corr_metric='simple',linkage_method='single'):
        if corr_metric == 'simple':
            corr = returns.corr()
        elif corr_metric == 'exponential':
            ewma_returns = returns.ewm(alpha= 0.94 ,adjust=False).mean()
            corr = ewma_returns.corr()
        elif corr_metric == 'noise':
            noise = np.random.normal(loc=0, scale=0.01, size=returns.shape)
            noisy_returns = returns + noise
            corr = noisy_returns.corr()
            
        d = np.sqrt(((1.-corr)/2.))
        d_tilde = scipy.spatial.distance.squareform(d,checks=False)
        clusters = scipy.cluster.hierarchy.linkage(d_tilde,linkage_method)
        
        return clusters,d,d_tilde
    
    def _get_quasi_diagonal_matrix(self):
        return scipy.cluster.hierarchy.to_tree(self.clusters, rd=False).pre_order()
    
    def plot_corr_matrix(self):
        
        sort_idx = self._get_quasi_diagonal_matrix()
        N = len(self.d)
        seriated_dist = np.zeros((N, N))
        a,b = np.triu_indices(N, k=1)
        seriated_dist[a,b] = self.d.values[[sort_idx[i] for i in a], [sort_idx[j] for j in b]]
        seriated_dist[b,a] = seriated_dist[a,b]
        
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        c1 = ax[0].pcolormesh(self.d, cmap='viridis', shading='auto')
        fig.colorbar(c1, ax=ax[0])
        ax[0].set_title('Original Order Distance Matrix', fontsize=12)
        ax[0].set_xlabel('Index')
        ax[0].set_ylabel('Index')

        c2 = ax[1].pcolormesh(seriated_dist, cmap='viridis', shading='auto')
        fig.colorbar(c2, ax=ax[1])
        ax[1].set_title('Re-ordered Distance Matrix', fontsize=12)
        ax[1].set_xlabel('Index')
        ax[1].set_ylabel('Index')

        plt.tight_layout()
        plt.show()
        
    def plot_weights_pie(self, hrp_w, n=10):

        hrp_series = pd.Series(hrp_w).sort_values(ascending=False)

        top_n = hrp_series.head(n)
        # top_n.plot.pie(figsize=(10, 10), autopct='%1.1f%%', cmap=f'tab{n}', legend=True)
        top_n.plot.pie(figsize=(10, 10), autopct='%1.1f%%', cmap=f'prism', legend=True)
        plt.legend().remove()

        plt.title(f'Top {n} HRP Portfolio Weights')
        plt.ylabel('')  # Hide y-axis label
        plt.show()
        
    def plot_dendogram(self,n = None ):
        fig, ax = plt.subplots(figsize=(10, 5))
        if n is None:
            scipy.cluster.hierarchy.dendrogram(self.clusters, \
                                           labels=list(self.returns.columns) , \
                                               ax=ax, orientation="top")
        else:
            rets = self.returns.loc[:,random.sample(list(self.returns.columns), n)]
            clusters_ = self._get_clusters(rets)[0]
            scipy.cluster.hierarchy.dendrogram(clusters_, \
                                           labels= list(rets.columns) , \
                                               ax=ax, orientation="top")
            
        ax.tick_params(axis="x", rotation=90) 
        plt.tight_layout()
        ax.set_title("Hierarchical Clustering Dendrogram")
        plt.show()

        
    def _computer_cluster_variance( self, cluster ):
        cov_cluster = self.cov.loc[cluster,cluster]
        weights = 1 / np.diag(cov_cluster)  
        weights /= weights.sum()
        return np.linalg.multi_dot((weights, cov_cluster, weights))
    
    def recursive_bisection( self, sorted_index):
    
        ordered_tickers =  self.cov.index[sorted_index].tolist()
        w = pd.Series(1., index=ordered_tickers)
        clusters = [ordered_tickers]

        while len(clusters) > 0:
            clusters = [item[j:k] for item in clusters for j,k in ((0, len(item) // 2), (len(item) // 2, len(item)))
                    if len(item) > 1 ]

            for k in range(0,len(clusters),2):
                cluster_k = clusters[k]
                cluster_l = clusters[k+1]

                #Definining variance 
                var_cluster_k = self._computer_cluster_variance(cluster_k)
                var_cluster_l = self._computer_cluster_variance(cluster_l)

                #Defining weights
                alpha = 1 - var_cluster_k / (var_cluster_k + var_cluster_l)
                w[cluster_k] *= alpha  
                w[cluster_l] *= 1 - alpha  

        w[np.abs(w) < 1e-4] = 0.
        w = np.round(w, 5)
        return collections.OrderedDict(w.sort_index())


    def calculate_hrp_weights(self,corr_metric='simple', linkage_method='single'):

        ## Step1
        self.clusters,self.d,_ = self._get_clusters(self.returns,corr_metric,linkage_method)
        
        ##Step2
        sorted_idx = self._get_quasi_diagonal_matrix()
        
        ##Step3
        w = self.recursive_bisection( sorted_idx )
        return w
    
    def calculate_MV_weights(self):
        inv_covar = np.linalg.inv(self.cov)
        u = np.ones(len(self.cov))

        w = np.dot(inv_covar, u) / np.dot(u, np.dot(inv_covar, u))
        w[np.abs(w) < 1e-4] = 0.
        w = np.round(w, 5)
        return collections.OrderedDict(zip(list(self.returns.columns) , w))


    def calculate_RP_weights(self):
        weights = (1 / np.diag(self.cov)) 

        w =  weights / sum(weights)
        w[np.abs(w) < 1e-4] = 0.
        w = np.round(w, 5)
        return collections.OrderedDict(zip(list(self.returns.columns) , w))

    # def calculate_unif_weights(self):

    #     w =  np.array([1 / len(self.cov) for i in range(len(self.cov))])
    #     w[np.abs(w) < 1e-4] = 0.
    #     w = np.round(w, 5)
    #     return collections.OrderedDict(zip(list(self.returns.columns) , w))
    