# Importamos librerias
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Importamos nuestro data frame
df = pd.read_csv('/Users/francoquintanilla/Desktop/advertising.csv')
df.head()

# Contamos si hay datos NaNs
df.isna().sum()

# Vemos el tamaño de los datos para separar por train y test
df.shape

# Visualizamos nuestras variables para poder hacer un mejor analisis dobre la regresión
fig, ax = plt.subplots(1, 3, figsize=(22,5))

ax[0].scatter(df['TV'], df['Sales'])
ax[0].set_title('"Contribución de la TV a las ventas"')

ax[1].scatter(df['Radio'], df['Sales'])
ax[1].set_title('"Contribución de la Radio a las ventas"')

ax[2].scatter(df['Newspaper'], df['Sales'])
ax[2].set_title('"Contribución del periodico a las ventas"')

plt.show()

# Vemos que la mejor opción sería irnos por la TV, por lo que
# dividimos nuestro data frame xs y nuestro goal (y) en est y train
x_train = df.iloc[0:160,0]
x_test = df.iloc[160:,0]

y_train = df.iloc[0:160,3]
y_test = df.iloc[160:,3]

# Vemos como quedarón las gráficas
fig, ax = plt.subplots(1, 2, figsize=(22,5))

ax[0].scatter(x_train, y_train)
ax[0].set_title('"Contribución de la TV a las ventas"')

ax[1].scatter(x_test, y_test)
ax[1].set_title('"Contribución de la Radio a las ventas"')

plt.show()

# Hipótesis (entrenamiento y error)
h = lambda x, theta: theta[0] + theta[1]*x
j_i = lambda x, y, theta: (h(x, theta) - y)**2 

# Parametros
alpha = 0.001
n = 100000
theta = [7, 0.05]

# Entrenamiento
for i in range(n):
  delta = []
  deltax = []

  for x_i, y_i in zip(x_train, y_train):
    delta.append(h(x_i, theta) - y_i)
    deltax.append((h(x_i, theta) - y_i)*x_i)

  # Thetas
  theta[0] = theta[0] - alpha/n*sum(delta)
  theta[1] = theta[1] - alpha/n*sum(deltax)

# Training 2.0
delta_train = []
for x_i, y_i in zip(x_train, y_train):
  delta_train.append(j_i(x_i, y_i, theta))  
 
J_train = 1/(2*n)*sum(delta_train)

# Validación
delta_test = []
for x_i, y_i in zip(x_test, y_test):
  delta_test.append(j_i(x_i, y_i, theta))  
 
J_test = 1/(2*n)*sum(delta_test)

print("Error en el train:", J_train)
print("Error en el test:", J_test)
print("Valores de las thetas", theta)

plt.scatter(df['TV'], df['Sales'])
plt.plot(x_test, theta[0] + theta[1]*x_test, 'r')
plt.title("Regresión Lineal sobre 'Contribución de la TV a las ventas'")
plt.xlabel("Distribución en TV")
plt.ylabel("Ventas")
plt.show()

# Podemos comparar con seaborn a ver que tan parecidos estamos.
sns.regplot(x='TV', y='Sales', data=df)
plt.title("Regresión Lineal sobre 'Contribución de la TV a las ventas' con Seaborn")
plt.xlabel("Distribución en TV")
plt.ylabel("Ventas")
plt.show()

# Hacemos algunas predicciones
y_pred = theta[0] + theta[1]*x_test

# Calculamos el coeficiente de determinación
c_mat = np.corrcoef(y_test, y_pred)
CoD = c_mat[0,1]
CoD = CoD**2
print("El coeficiente de determinación es de:", CoD)

# Calculamos el MSE (Mean Squared Error)
MSE = np.square(np.subtract(y_test, y_pred)).mean()
print("El error cuadratico medio es de:", MSE)

""" Como podemos ver, vamos por buen camino de la regresión lineal de primer orden con un coeficiente de determinación de 0.798, 
el cual se podría mejorar y ajustar con un modelo de regresión no lineal. 

Si tratamos de meter las variables de "Radio" y "Newspaper" el modelo empieza a tener errores 
demasiado grandes y el modelo ya no se ajusta correctamente, ya que estas variables meten demasiado ruido a nuestro probelma original."""
