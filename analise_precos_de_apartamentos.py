import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
'''from sklearn.preprocessing import OrdinalEncoder'''
'''from sklearn.preprocessing import OneHotEncoder'''
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#px.set_mapbox_access_token("Default public token")
px.set_mapbox_access_token(open("mapbox_token.txt").read())

df_data = pd.read_csv("sao-paulo-properties-april-2019.csv")

df_rent = df_data[df_data["Negotiation Type"] == 'rent']

fig = px.scatter_mapbox(df_rent, lat ="Latitude", lon = "Longitude", color = "Price", size = "Size", color_continuous_scale = px.colors.cyclical.IceFire, size_max=15, zoom=10, opacity=0.4)

#mudar a cor para ficar mais visível
fig.update_coloraxes(colorscale = [[0, 'rgb(166, 206, 227, 0.5)'],
                                   [0.02, 'rgb(31, 120, 180, 0.5)'],
                                   [0.05, 'rgb(178, 223, 138, 0.5)'],
                                   [0.10, 'rgb(51, 160, 44, 0.5)'],
                                   [0.15, 'rgb(251, 154, 153, 0.5)'],
                                   [1, 'rgb(227, 26, 28, 0.5)'],
                                   ],
                                   )

#delimitar tamanho do gráfico
fig.update_layout(height = 800, mapbox = dict(center = go.layout.mapbox.Center(lat=-23.543138, lon=-46.69486)))

#descrição de cada coluna
'''print(df_rent.describe())
'''
#mostrar histogramas sobre cada coluna
'''df_rent.hist(bins=30, figsize=(30, 15))
plt.tight_layout()
plt.show()'''

#mostrar a correlação
'''print(df_rent.corr(numeric_only=True))'''

#mostrar a correlação de forma organizada para a variável Price
'''print(df_rent.corr(numeric_only=True)["Price"].sort_values(ascending=False))'''

#root mean squared error

#limpeza de dados (retirando colunas inuteis)
df_cleaned = df_rent.drop(["New", "Property Type", "Negotiation Type"], axis=  1)

#nesse momento vamos numerar as informações da coluna distritos, pois como está como dados de string, o 
#python não consegue trabalhar com eles, assim vamos transformar cada valor em um numero
#Mas esse ordinaleconder só é bom quando temos poucas classes, pois do jeito que esta agora, o python pode acabar
#se confundindo e achar que essa grande quantidade de numeros tem relação com algo. 
'''ordinal_encoder = OrdinalEncoder()

district_encoder = ordinal_encoder.fit_transform(df_rent[["District"]])
'''

#Portanto, vamos experimentar utilizar a biblioteca chamada OneHOtEncoder, para criar colunas para cada distrito e assim
#caso o apartamento pertencer a determinado distrito a coluna marcara com 1 para sim e 0 para não.
#porem ainda não é a melhor forma.

'''cat_encoder = OneHotEncoder()

housing_cat_1hot = cat_encoder.fit_transform(df_cleaned[["District"]])
print(housing_cat_1hot)'''

#a melhor forma para fazer isso é usando uma biblioteca do pandas chamada get_dummies

one_hot = pd.get_dummies(df_cleaned["District"], dtype=int)

df = df_cleaned.drop("District", axis=1)
df = df.join(one_hot)

#TREINAMENTO DE MODELOS
# devemos importar o from sklearn.model_selection import train_test_split
#devemos criar as variaveis de dados de treino e dados de teste
#aqui eu vou definir o y como a variavel Price e x será tudo menos o Price
y = df["Price"]
x = df.loc[:, df.columns != "Price"]

x_train, x_teste, y_train, y_teste = train_test_split(x,y, test_size=0.3)

#MODELO DE REGRESSÃO LINEAR (LINEAR REGRESSION)
#vamos testar alguns modelos ainda nos dados de treino para saber qual seria melhor para só depois aplicar
#nos dados de teste
# fazer o import de from sklearn.linear_model import LinearRegression

#aqui nós vamos instanciar essa classe
lin_reg = LinearRegression()

#aqui ele vai utilizar a função custo para fazer a otimização do modelo
lin_reg.fit(x_train, y_train)

#aqui já podemos fazer alguns TESTES para ver como esse modelo está se saindo:
'''LinearRegression()
a = x_train.iloc[:5]
b = y_train.iloc[:5]

print("Predições", lin_reg.predict(a))
print("Labels", b.values)'''

#COMO MEDIR COMO A REGRESSÃO LINEAR PERFORMOU?
#vamos usar o from sklearn.metrics import mean_squared_error
# ele nos trará um valor que será o quanto o modelo está errando no valor do apartamento

'''preds = lin_reg.predict(x_train)
lin_mse = mean_squared_error(y_train, preds)

lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)'''

#aqui ele volta o valor 1990.63416155939 que seria o tanto que ele está errando pra mais e pra menos em reais.


#MODELO DE REGRESSÃO DE ÁRVORE DE DECISÃO (DECISION TREE REGRESSOR)

#instanciar a classse
'''tree_reg = DecisionTreeRegressor()
tree_reg.fit(x_train, y_train)

preds = tree_reg.predict(x_train)
tree_mse = mean_squared_error(y_train, preds)

tree_rmse = np.sqrt(tree_mse)
print(tree_rmse)'''

#quando usamos esse modelo o resultado é de 27.43676833805326; bem diferente ao da regressão linear

#TÉCNICA DO CROSS VALIDATION (NÃO SE ESQUECER DOS 30% PARA TESTE E 70% PARA TREINO) 
# essa técnica só usa os dados de treino
# usamos o from sklearn.model_selection import cross_val_score
'''
scores =  cross_val_score(tree_reg, x_train, y_train, scoring="neg_mean_squared_error", cv= 10)'''
#o scoring é uma métrica de utilidade e no geral são contrário à metrica de custo.
#nós sempre tentamos maximizar as metricas de utilidade e minimizar as metricas de custo

'''tree_rmse_scores = np.sqrt(-scores)'''

#criar função para printar os scores, as medias dos scores e o desvio padrão

'''def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standart deviation:", scores.std())

print(display_scores(tree_rmse_scores))'''

#fato interessante: ao usar o modelo de arvore de decisao e pedir para ele trazer a media dos erros quadraticos
#de dados que ele "viu", ele retornar um BOM valor de 27.43676833805326. Mas quando usamos um valor que ele 
#"não viu" ele retorna um valor de 2280.573267887047, que é mais alto que o valor da regressão linear.


#AQUI VAMOS FAZER O MESMO TESTE DE CROSS VALIDATION COM O MODELO DE REGRESSÃO LINEAR:

'''lscores =  cross_val_score(lin_reg, x_train, y_train, scoring="neg_mean_squared_error", cv= 10)
lin_rmse_scores = np.sqrt(-lscores)

def display_lscores(lscores):
    print("Scores:", lscores)
    print("Mean:", lscores.mean())
    print("Standart deviation:", lscores.std())

print(display_lscores(lin_rmse_scores))'''

#aqui temos uma clara diferença de que o modelo de regressão linear responde melhor do que o modelo de 
#árvore de decisão; o modelo de regressão linear mostrou medias de: 1933.3848793889672. Enquanto o modelo
#de árvore de decisão mostou medias de: 2354.748247195957


#ÚLTIMO MODELO: FLORESTAS ALEATÓRIAS (RANDOM FORESTS)
#importar o from sklearn.ensemble import RandomForestRegressor

'''rf_reg = RandomForestRegressor()
rf_reg.fit(x_train, y_train)

preds = rf_reg.predict(x_train)
rf_mse = mean_squared_error(y_train, preds)

rf_rmse = np.sqrt(rf_mse)
print(rf_rmse)'''

#já de inicio o modelo de random forests foi melhor, mostrando uma média de 641.6240270323702
#agora vamos fazer um cross validation com esse modelo:

'''scores =  cross_val_score(rf_reg, x_train, y_train, scoring="neg_mean_squared_error", cv= 10)
rf_rmse_scores = np.sqrt(-scores)

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standart deviation:", scores.std())

print(display_scores(rf_rmse_scores))'''

#nesse momento é normal demorar para aparecer o restultado, tendo em vista que ele é mais complexo, mas em compensação
#ele traz um resultado bem melhor, com uma media de: Mean: 1789.8637355765338, a melhor dentre todos os modelos até então.



#AVALIAR E OTIMIZAR O MODELO
#aqui vamos importar o from sklearn.model_selection import GridSearchCV


#essa classe pede que a gente passe uma lista com um dicionário contendo os parâmetros que eu quero atualizar
#e uma lista contendo todos os parametros que eu quero testar. ao todo serão 18 modelos diferentes (3x4(12) do primeiro + 
#1x2x3(6) do segundo)

param_grid = [
    {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
    {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]}
]

forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring="neg_mean_squared_error",
                           return_train_score=True)

grid_search.fit(x_train, y_train)

#aqui ele vai me falar qual foi o melhor parametro, retornando: 
#{'max_features': 6, 'n_estimators': 30}
print(grid_search.best_params_)

#ele automaticamente salva uma variavel para mostrar o melhor estimator que seria:
#grid_search.best_estimator_
# e mostra o seguinte: RandomForestRegressor(max_features=6, n_estimators=30)

print(grid_search.best_estimator_)



#AGORA PARA SABER SE O MODELO É BOM, PODEMOS COMPARAR COM OS DADOS DE TESTE TOTAL

final_model = grid_search.best_estimator_
final_model_prediciton = final_model.predict(x_teste)


#aqui ele vai fazer a comparação e trará o resultado: 1809.223784457667
final_mse = mean_squared_error(y_teste, final_model_prediciton)
print(np.sqrt(final_mse))


#vamos ver em um gráfico:

teste = go.Figure(data=[go.Scatter(y=y_teste.values),
                        go.Scatter(y=final_model_prediciton)])

teste.show()

#O modelo conseguiu chegar a valores proximos

#mostrar o mapa com os dados
'''fig.show()'''


