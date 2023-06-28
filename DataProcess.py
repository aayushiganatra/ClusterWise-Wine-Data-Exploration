import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
class ClusterProcess:
    def __init__(self,n_pca_components=2):
        self.n_pca_comp=n_pca_components
        self.pca=PCA(n_components=n_pca_components)
        self.normalize_std_scaler=StandardScaler()
    def readData(self,name):
        return pd.read_csv(name+".csv")
    def findPCA(self,data):
        pc=self.pca.fit_transform(data)
        pc_df=pd.DataFrame(data=pc,columns=['comp_'+str(i) for i in range(1,self.n_pca_comp+1)])
        return pc_df
    def do_clustering(self,data,n_cluster,method):
        if method.__name__==AgglomerativeClustering.__name__:
            cluster_algo=method(n_clusters=n_cluster)
            return cluster_algo.fit_predict(data),cluster_algo.labels_,silhouette_score(data,cluster_algo.labels_)
        elif method.__name__==KMeans.__name__:
            cluster_algo=method(n_clusters=n_cluster)
            cluster_algo.fit(data)
            return cluster_algo.predict(data),cluster_algo.cluster_centers_,cluster_algo.labels_,silhouette_score(data,cluster_algo.labels_),cluster_algo.inertia_
        else:
            cluster_algo=method(n_clusters=n_cluster)
            cluster_algo.fit(data)
            return cluster_algo.predict(data),cluster_algo.cluster_centers_,cluster_algo.labels_,silhouette_score(data,cluster_algo.labels_),cluster_algo.inertia_
    def fitScale(self,data):
        self.normalize_std_scaler.fit(data)
    def tranScale(self,data):
        return self.normalize_std_scaler.transform(data)
    def invScale(self,data):
        return self.normalize_std_scaler.inverse_transform(data)
    def plot3dScatter(self,data,color='None'):
        if color=='None':
            fig=px.scatter_3d(data,x='comp_1',y='comp_2',z='comp_3',size_max=10)
        else:
            fig=px.scatter_3d(data,x='comp_1',y='comp_2',z='comp_3',color=color,size_max=10)
        fig.update_traces(marker_size = 3)
        fig.show()
    def plotBar(self,data,column_names,color='None'):
        if color=='None':
            fig = px.bar(pd.DataFrame(data=data,columns=column_names), x=column_names[0],y=column_names[1])
        else:
            fig = px.bar(pd.DataFrame(data=data,columns=column_names), x=column_names[0],y=column_names[1],color=color)
        fig.show()