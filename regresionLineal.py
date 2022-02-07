# Tratamiento de datos
# ==============================================================================
from matplotlib.axes import Axes
import pandas as pd
import numpy as np
# Gráficos
# ==============================================================================
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

# Conexión a la base de datos
import sqlalchemy

# Preprocesado y modelado
# ==============================================================================
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sqlalchemy import false

# Configuración matplotlib
# ==============================================================================
plt.rcParams['image.cmap'] = "bwr"
plt.rcParams['figure.dpi'] = "100"
plt.rcParams['savefig.bbox'] = "tight"
style.use('ggplot') or plt.style.use('ggplot')

# Configuración warnings
# ==============================================================================
import warnings
warnings.filterwarnings('ignore')

#Conexion a la base de datos 
# ===============================================================================
path = "sqlite:///Beisbol.db"
motorDB = sqlalchemy.create_engine(path)
conectarDB = motorDB.connect()

# consulta de sql 
query = '''SELECT e.nombres AS equipos, b.tiros AS bateos,r.tiros_run AS runs
        FROM equipos e
        INNER JOIN 
        bateos b ON b.id_equi = e.id_equipos
        INNER JOIN 
        runs r ON r.id_bat = b.id_bateos;'''

result = conectarDB.execute(query)

# Creación del dataSet
datos = pd.DataFrame(result.fetchall())
datos.columns = result.keys()
print(datos);


# Gráfico
# ==============================================================================
fig, ax = plt.subplots(figsize=(6, 3.84))
datos.plot(
x = 'bateos',
y = 'runs',
c = 'firebrick',
kind = "scatter",
ax = ax
)
ax.set_title('Distribución de bateos y runs');
plt.show()


# Correlación lineal entre las dos variables
# ==============================================================================
corr_test = pearsonr(x = datos['bateos'], y = datos['runs'])
print("Coeficiente de correlación de Pearson: ", corr_test[0])
print("P-value: ", corr_test[1])


# División de los datos en train y test
# ==============================================================================
X = datos[['bateos']]
y = datos['runs']
X_train, X_test, y_train, y_test = train_test_split(
X.values.reshape(-1,1),
y.values.reshape(-1,1),
train_size = 0.8,
random_state = 1234,
shuffle = True
)
# Creación del modelo
# ==============================================================================
modelo = LinearRegression()
modelo.fit(X = X_train.reshape(-1, 1), y = y_train)

# Información del modelo
# ==============================================================================
print("Intercept:", modelo.intercept_)
print("Coeficiente:", list(zip(X.columns, modelo.coef_.flatten(), )))
print("Coeficiente de determinación R^2:", modelo.score(X, y))

# Error de test del modelo
# ==============================================================================
predicciones = modelo.predict(X = X_test)
print(predicciones[0:3,])
rmse = mean_squared_error(
y_true = y_test,
y_pred = predicciones,
squared = False
)
print("")
print(f"El error (rmse) de test es: {rmse}")


