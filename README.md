# Modulo2-ML

En este entregable se hace la implementación de una técnica de aprendizaje máquina (ML) sin el uso de alguna libreria de aprendizaje máquina. En este caso se optó por usar la Regresión Lineal en base a las funciones lambda, tanto para las pruebas de hipótesis como las funciones de costo. 

El dataset utilizado se encuentra en: https://www.kaggle.com/code/ashydv/sales-prediction-simple-linear-regression/notebook
de la pagina de kaggle, titulado como "Sales Prediction".

Para este caso, despues de visualizar el comportamiento de los datos, utilizamos la regresión lineal simple en base a la Contribución de la TV a las ventas, ya que las demás contribuciones no siguien una linea de tendencia tan significativa. 

Para poder hacer el analisis de nuestra técnica, dividimos nuestro dataset en 80% datos para entrenamiento y 20% datos para la evaluación.

Para los casos iniciales, despues de un poco de investigación, y de estar a prueba y error, los parametros iniciales fuerón los siguientes para nuestro modelo:

* alpha = 0.001
* theta = [7, 0.05]

En donde nuestro modelo trata de buscar el mejor "fit" para nuestros datos del modelo. 

Consecuentemente, graficamos la linea de tendencia que sigue nuestro modelo de Regresión Lineal, junto a los datos de nuestro dataset para que la visualización de la tendencia fuera más clara, junto con la regresión lineal del paquete de seaborn, para ver si tienen similitudes, las cuales si las tienen. 

Por ultimo, hicimos algunas prediciones con los valores de la ecuación de nuestro modelo con el 20% de los datos de la evaluación, para despues poder sacar los valores de:

* Coeficiente de determinación: 0.811
* Error cuadratico medio: 5.762

## Analizando los datos 

Después de hacer el analisis de los datos obtivimos los siguientes resultados.

Para el sesgo, graficamos los datos del dataset, en donde el comportamiento es de la siguiente manera.

<img width="406" alt="Captura de Pantalla 2022-09-08 a la(s) 14 55 38" src="https://user-images.githubusercontent.com/111082680/189213954-bb919afc-b702-4a0b-a9eb-c39407fbde48.png">

Tambien, graficamos el comportamiento de nuestras predicciones.

<img width="412" alt="Captura de Pantalla 2022-09-08 a la(s) 14 50 23" src="https://user-images.githubusercontent.com/111082680/189214113-19713b52-6ace-4343-8409-e57f38e4fe84.png">




















