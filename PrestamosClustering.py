import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
import plotly.express as px

# Initialize LabelEncoder and StandardScaler
scaler = StandardScaler()

def load_data(file):
    df = pd.read_excel(file)

    # Fill missing values with mean of respective columns for numeric data
    df_numeric = df.select_dtypes(include=[np.number])
    df[df_numeric.columns] = df_numeric.fillna(df_numeric.mean())

    # Fill missing values with mode for categorical data
    df_categorical = df.select_dtypes(exclude=[np.number])
    for column in df_categorical.columns:
        df[column].fillna(df[column].mode()[0], inplace=True)

    # Create a copy of df before transformations for dropdown options
    df_original = df.copy()

    categorical_features = ['profesion (groups)', 'Tipo de cliente (groups)', 'Provincia', 'Ciudad', 'Nacionalidad']

    encoders = {}
    for feature in categorical_features:
        le = LabelEncoder()
        df[feature] = le.fit_transform(df[feature].astype(str))
        encoders[feature] = le

    # Normalize the data
    df = pd.DataFrame(scaler.fit_transform(df), columns = df.columns)

    # Determine the optimal number of clusters using the Elbow method
    wcss = []  # Within-Cluster-Sum-of-Squares
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(df)
        wcss.append(kmeans.inertia_)

    n_clusters = wcss.index(min(wcss)) + 1
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
    df['Cluster'] = kmeans.fit_predict(df)

    return df, kmeans, categorical_features, df_original, encoders

def app():
    st.title("Client Behavior Analysis")

    # Load the data
    file = st.file_uploader("Please upload your data", type=['xlsx', 'xls'])
    if file is not None:
        df, kmeans, categorical_features, df_original, encoders = load_data(file)

        st.write("Please input the client's information below:")

        # Collect user input
        profesion = st.selectbox('Profesion', df_original['profesion (groups)'].unique())
        tipo_de_cliente = st.selectbox('Tipo de Cliente', df_original['Tipo de cliente (groups)'].unique())
        edad = st.number_input('Edad', value=20)
        provincia = st.selectbox('Provincia', df_original['Provincia'].unique())
        ciudad = st.selectbox('Ciudad', df_original['Ciudad'].unique())
        nacionalidad = st.selectbox('Nacionalidad', df_original['Nacionalidad'].unique())
        count_of_id_prestamo = st.number_input('Count of ID prestamo', value=0)
        average_of_monto_aprobado = st.number_input('Average of Monto aprobado', value=0.0)
        count_of_codigo_cliente = st.number_input('Count of Codigo cliente', value=0)
        average_of_tasa_de_prestamo = st.number_input('Average of Tasa de Préstamo', value=0.0)
        average_of_plazo = st.number_input('Average of Plazo', value=0)

        # Create DataFrame from user input
        new_client = pd.DataFrame([[profesion, tipo_de_cliente, edad, provincia, ciudad, nacionalidad, count_of_id_prestamo, average_of_monto_aprobado, count_of_codigo_cliente, average_of_tasa_de_prestamo, average_of_plazo]], columns=df.columns[:-1])

        # Preprocess the new client in the same way as the original data
        for feature in categorical_features:
            if new_client[feature][0] in encoders[feature].classes_:
                new_client[feature] = encoders[feature].transform(new_client[feature])
            else:
                st.error(f'Error: {feature} input "{new_client[feature][0]}" is not recognized. Please try again.')
                return

        new_client = pd.DataFrame(scaler.transform(new_client), columns=new_client.columns)

        # Predict the cluster for the new client
        new_client_cluster = kmeans.predict(new_client)

        st.write(f"The new client belongs to cluster: {new_client_cluster[0]}")

        # Calculate cluster sizes
        cluster_sizes = df['Cluster'].value_counts(normalize=True)

        # Check if the new client belongs to a "weird" cluster
        if cluster_sizes[new_client_cluster[0]] < 0.10:
            st.write("Warning: This client is behaving in a potentially weird way.")

        # Visualize the clusters
        fig = px.scatter(df, x=df.columns[0], y=df.columns[1], color='Cluster')
        st.plotly_chart(fig)

if __name__ == "__main__":
    app()
