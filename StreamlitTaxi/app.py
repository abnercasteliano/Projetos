import pandas as pd
import numpy as np
import streamlit as st

# Definindo títulos horizontal - beta_container
header = st.beta_container()
dataset = st.beta_container()
features = st.beta_container()
training = st.beta_container()

# definindo o título
with header:
    st.title('Bem vindo ao projeto de Data Science!')
    st.text('Neste projeto podemos analisar o trânsito de táxis em Nova Iorque.')
    st.text('O objetivo do projeto é calcular a distância total da corrida de cada passageiro.')

# definindo os subtítulos

with dataset:
    st.header('NYC taxi data')
    st.text('O dataset foi coletado no site do governo do estado de Nova Iorque. Ele contém os dados '
            'da '
            'corrida de taxi de milhares de passageiros na cidade de Nova Iorque.')

    taxi_data = pd.read_csv('taxi_data.csv')
    st.dataframe(taxi_data.head(10))

    # plotando a distribuição dos pontos de partida
    st.subheader('Distribuição do ID dos pontos de partida')
    pulocation_dist = pd.DataFrame(taxi_data['PULocationID'].value_counts().head(50))
    st.bar_chart(pulocation_dist)

with features:
    st.header('Variáveis criadas')

    taxi_data['median_total_amount'] = np.median(taxi_data['total_amount'])
    st.markdown('* **median_total_amount:** Esta variável foi criada utilizando-se a mediana do pagamento total final, '
                'com o objetivo de calcular o preço médio pago pelos passageiros.')

with training:
    st.header('Treinando o modelo')
    st.text('Aqui você pode selecionar algumas variáveis e analisar a performance do modelo!')

    st.text('Lista de variáveis do modelo:')
    st.table(taxi_data.drop('trip_distance', axis=1).columns)

    sel_col, display_col = st.beta_columns(2)

    max_depth = sel_col.slider('Qual deve ser número máximo de variáveis do modelo?', min_value=5, max_value=15, value=5, step=5)

    input_feature = sel_col.text_input('Digite uma variável de entrada', 'PULocationID')

    n_estimators = sel_col.selectbox('Quantas árvores o algoritmo deve ter?', options=[64, 84, 100, 'No limit'], index=0)

    # Seleção e teste do modelo
    from sklearn.ensemble import RandomForestRegressor

    if n_estimators == 'No limit':
        regr = RandomForestRegressor(max_depth=max_depth, n_estimators=128)
    else:
        regr = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)

    # divisão dos dados
    X = taxi_data[[input_feature]].values
    y = taxi_data[['trip_distance']].values

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    regr.fit(X_train, y_train.ravel())
    y_prediction = regr.predict(X_test)

    # demonstração da performance do modelo
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import r2_score

    if st.button('Calcular'):
        display_col.subheader('O erro médio quadrado do modelo é: ')
        display_col.write(mean_absolute_error(y_test, y_prediction))

        display_col.subheader('O erro médio quadrado do modelo é: ')
        display_col.write(mean_squared_error(y_test, y_prediction))

        display_col.subheader('O R²_score do modelo é: ')
        display_col.write(r2_score(y_test, y_prediction))
