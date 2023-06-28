import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance

# Initialize StandardScaler
scaler = StandardScaler()

def load_data(file):
    df = pd.read_excel(file)

    # Fill missing values with mean of respective columns for numeric data
    df_numeric = df.select_dtypes(include=[np.number])
    df[df_numeric.columns] = df_numeric.fillna(df_numeric.mean())

    # Normalize the data
    df_norm = pd.DataFrame(scaler.fit_transform(df.drop(columns=['Código del Cliente'])), 
                           columns = df.drop(columns=['Código del Cliente']).columns)
    df_norm['Código del Cliente'] = df['Código del Cliente'].values

    # Determine the optimal number of clusters using the Elbow method
    wcss = []  # Within-Cluster-Sum-of-Squares
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(df_norm.drop(columns=['Código del Cliente']))
        wcss.append(kmeans.inertia_)

    n_clusters = wcss.index(min(wcss)) + 1
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
    df_norm['Cluster'] = kmeans.fit_predict(df_norm.drop(columns=['Código del Cliente']))

    return df_norm, n_clusters, kmeans

def get_weird_clients(df, kmeans):
    df_original = df.copy()  # copy the dataframe
    df_original.drop(columns=['Cluster'], inplace=True)  # drop the Cluster column

    distances = []
    for i in range(kmeans.n_clusters):
        cluster_points = df[df['Cluster'] == i].drop(columns=['Cluster', 'Código del Cliente'])
        centroid = kmeans.cluster_centers_[i]
        distances.extend(distance.cdist([centroid], cluster_points, 'euclidean')[0])

    df_original['Distance'] = distances
    weird_clients = df_original[df_original['Distance'] > df_original['Distance'].mean() + df_original['Distance'].std()]['Código del Cliente'].tolist()
    return weird_clients

def app():
    st.title("Análisis del comportamiento del cliente Certificados")

    pin = st.text_input("Please input your PIN", type="password")

    if pin == "0000":

        # Load the data
        file = st.file_uploader("Please upload your data", type=['xlsx', 'xls'])
        if file is not None:
            df, n_clusters, kmeans = load_data(file)

            # Display the list of weird clients
            weird_clients = get_weird_clients(df, kmeans)
            st.write("List of Clients Behaving Weirdly:")
            st.write(weird_clients)

            # Display the number of clusters found
            st.write(f"Number of clusters found: {n_clusters}")

            # Display the contents of each cluster
            for i in range(n_clusters):
                clients_in_cluster = df[df['Cluster'] == i]['Código del Cliente'].tolist()
                st.write(f"Clients in cluster {i}:")
                st.write(clients_in_cluster)

        else:
            st.error("You haven't uploaded a file.")
    else:
        st.error("Incorrect PIN. Please try again.")

if __name__ == "__main__":
    app()
