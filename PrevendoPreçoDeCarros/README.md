# Previsão do preço de carros

O DataSet 'carprices.csv' contém os dados de três modelos de carros, a quilometragem percorrida, os anos de uso e o preço de venda.
O objetivo é implementar um algoritmo de Machine Learning adequado e responder às questões de negócio:

1)	Prever o preço de um Mercedez Benz com 4 anos de uso e quilometragem rodada de 45.000 KM.
2)	Prever o preço de uma BMW X5 com 7 anos de uso e quilometragem de 86.000 KM.
3)	Me diga qual a precisão do modelo em porcentagem %.

Inicialmente foi feita uma análise para verificar a correlação entre as variáveis. Verificou-se as variáveis mais relevantes ("Mileage" e "Years").
Após categorizar os dados utilizando o método dummies, verificou-se que todas as variáveis("Car Model", "Mileage" e "Years") influenciam de alguma maneira a variável alvo("Sell Price").

Com isso, observou-se que o proplema trata-se de uma Regressão Linear de multíplas variáveis. O melhor modelo testado e escolhido foi o Linear Regression.

### Respostas Finais:

Mercedez Benz C class com 4 anos de uso e 45.000 KM rodados: preço de venda R$ 34.778,40

BMW X5 com 7 anos de uso e 86.000 KM rodados: preço de venda R$ 17.630,50

Acurácia do modelo de Machine Learning: 85%