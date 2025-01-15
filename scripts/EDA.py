import os
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

def plot_geographical_distribution(housing, output_path):
    # Gera um gráfico de dispersão geográfico baseado em longitude e latitude,
    # com tamanho proporcional à população e cor representando o valor médio das casas.
    housing.plot(
        kind="scatter",
        x="longitude",
        y="latitude",
        alpha=0.4,
        s=housing["population"] / 100,
        label="Population",
        c="median_house_value",
        cmap=plt.get_cmap("jet"),
        colorbar=True,
    )
    plt.title("Distribuição Geográfica das Casas")
    plt.savefig(os.path.join(output_path, "geographical_distribution.png"))  # Salva no diretório fornecido
    plt.close()

    plt.legend()
    
    # Salva o gráfico na pasta de saída
    output_file = os.path.join(output_path, "distribuicao_geografica.png")
    plt.savefig(output_file)
    plt.close() 

def plot_scatter_matrix(housing, output_path, attributes=None):
    # Plota uma matriz de dispersão para visualizar correlações entre os atributos especificados.
    
    if attributes is None:
        attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
    scatter_matrix(housing[attributes], figsize=(12, 8))
    
    # Salva o gráfico da matriz de dispersão
    output_file = os.path.join(output_path, "scatter_matrix.png")
    plt.savefig(output_file)
    plt.close()  # Fecha a figura para liberar memória

def prepare_data(housing):
    # Prepara os dados removendo a coluna 'median_house_value' e separando os rótulos.
    # Retorna os dados de entrada (housing) e os rótulos (housing_labels).
    
    housing_prepared = housing.drop("median_house_value", axis=1)
    housing_labels = housing["median_house_value"].copy()
    return housing_prepared, housing_labels
def plot_geographical_distribution(housing, output_path):
    # Gera um gráfico de dispersão geográfico baseado em longitude e latitude,
    # com tamanho proporcional à população e cor representando o valor médio das casas.
    
    housing.plot(
        kind="scatter",
        x="longitude",
        y="latitude",
        alpha=0.4,
        s=housing["population"] / 100,
        label="população",
        figsize=(10, 7),
        c="median_house_value",
        cmap=plt.get_cmap("jet"),
        colorbar=True,
    )
    plt.legend()
    
    # Salva o gráfico na pasta de saída
    output_file = os.path.join(output_path, "distribuicao_geografica.png")
    plt.savefig(output_file)
    plt.close()  # Fecha a figura para liberar memória

def plot_scatter_matrix(housing, output_path, attributes=None):
    # Plota uma matriz de dispersão para visualizar correlações entre os atributos especificados.
    
    if attributes is None:
        attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
    scatter_matrix(housing[attributes], figsize=(12, 8))
    
    # Salva o gráfico da matriz de dispersão
    output_file = os.path.join(output_path, "scatter_matrix.png")
    plt.savefig(output_file)
    plt.close()  # Fecha a figura para liberar memória

def prepare_data(housing):
    # Prepara os dados removendo a coluna 'median_house_value' e separando os rótulos.
    # Retorna os dados de entrada (housing) e os rótulos (housing_labels).
    
    housing_prepared = housing.drop("median_house_value", axis=1)
    housing_labels = housing["median_house_value"].copy()
    return housing_prepared, housing_labels