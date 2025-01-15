import os
import tarfile
import urllib.request
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from zlib import crc32
import matplotlib.pyplot as plt

import os
import tarfile
import urllib.request
import pandas as pd

# Definir os caminhos
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Caminho para a raiz do projeto
HOUSING_PATH = os.path.join(BASE_DIR, "datasets", "housing")
HOUSING_URL = "https://github.com/ageron/handson-ml2/raw/master/datasets/housing/housing.tgz"
OUTPUT_PATH = os.path.join(BASE_DIR, "outputs")

# Função para baixar e extrair os dados
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    with tarfile.open(tgz_path) as housing_tgz:
        housing_tgz.extractall(path=housing_path)

# Função para carregar os dados
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    if not os.path.exists(csv_path):
        print(f"Arquivo não encontrado: {csv_path}. Baixando agora...")
        fetch_housing_data()  # Baixar os dados, caso o arquivo não exista
    housing = pd.read_csv(csv_path)
    
    # Adicionar a coluna 'id' com o índice
    housing['id'] = housing.index
    
    return housing


    # Certifique-se de que os dados estão carregados
housing = load_housing_data()
print("Dados carregados com sucesso!")


# Função para adicionar uma coluna 'id' se não existir
def add_id_column(housing):
    if "id" not in housing.columns:
        housing["id"] = housing.index
    return housing


# Função para dividir o conjunto de dados em treino e teste
def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

# Carregar os dados
housing = load_housing_data()

# Dividir o dataset em conjunto de treino e teste
train_set, test_set = split_train_test_by_id(housing, test_ratio=0.2, id_column="id")

# Divisão aleatória para controle
random_train_set, random_test_set = train_test_split(housing, test_size=0.2, random_state=42)

# Criar uma coluna de categorias de renda
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

# Plotar a distribuição de renda
housing["income_cat"].hist()
output_file = os.path.join(OUTPUT_PATH, "distribuicao_renda.png")
plt.savefig(output_file)
plt.close()  # Fecha a figura após salvar o gráfico

# Aplicar a divisão estratificada com base na renda
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# Remover a coluna de categorias de renda
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

# Funções de codificação e transformação
def encode_ordinal(housing_cat):
    ordinal_encoder = OrdinalEncoder()
    housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
    
    # Salvar os resultados da codificação ordinal
    output_file = os.path.join(OUTPUT_PATH, "housing_cat_encoded_ordinal.csv")
    housing_cat_encoded_df = pd.DataFrame(housing_cat_encoded, columns=housing_cat.columns)
    housing_cat_encoded_df.to_csv(output_file, index=False)
    
    return housing_cat_encoded, ordinal_encoder.categories_

def encode_one_hot(housing_cat):
    cat_encoder = OneHotEncoder()
    housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
    
    # Salvar os resultados da codificação One-Hot
    output_file = os.path.join(OUTPUT_PATH, "housing_cat_1hot.csv")
    housing_cat_1hot_df = pd.DataFrame(housing_cat_1hot.toarray(), columns=cat_encoder.get_feature_names_out())
    housing_cat_1hot_df.to_csv(output_file, index=False)
    
    return housing_cat_1hot

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6
        rooms_per_household = x[:, rooms_ix] / x[:, households_ix]
        population_per_household = x[:, population_ix] / x[:, households_ix]
        
        if self.add_bedrooms_per_room:
            bedrooms_per_room = x[:, bedrooms_ix] / x[:, rooms_ix]
            transformed_data = np.c_[x, rooms_per_household, population_per_household, bedrooms_per_room]
            columns = list(range(x.shape[1])) + ["rooms_per_household", "population_per_household", "bedrooms_per_room"]
        else:
            transformed_data = np.c_[x, rooms_per_household, population_per_household]
            columns = list(range(x.shape[1])) + ["rooms_per_household", "population_per_household"]

        # Salvar os dados transformados
        output_file = os.path.join(OUTPUT_PATH, "transformed_data.csv")
        transformed_data_df = pd.DataFrame(transformed_data, columns=columns)
        transformed_data_df.to_csv(output_file, index=False)
        
        return transformed_data

