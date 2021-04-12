---
layout: default
title: Comparación de Algoritmos
nav_order: 14
---

# Comparación de Algoritmos
{: .fs-10 .no_toc }

El **objetivo** de la unidad es conocer y aplicar diferentes procedimientos estadísticos para comparar y conocer el rendimiento de algoritmos. 

## Tabla de Contenido
{: .no_toc .text-delta }

1. TOC
{:toc}

---

# Introducción

Hasta el momento se han descrito diferentes algoritmos de clasificación y
regresión; se han presentado diferentes medidas para conocer su rendimiento,
pero se ha dejado de lado el conocer la distribución de estas medidas
para poder tener mayor información sobre el rendimiento del algoritmo 
y también poder comparar y seleccionar el algoritmo que tenga las mejores
prestaciones ya sea en rendimiento o en complejidad. 

# Intervalos de confianza

El análisis del rendimiento se inicia partiendo de que el rendimiento 
se puede estimar a partir del conjunto de prueba, por facilidad, a 
esta estimación la denominaremos $$\hat \theta_N$$ y este valor estima
el rendimiento real $$\theta$$, el cual se conoce pero no es una variable
aleatoria dado que es constante. En generar un intervalo de confianza
para $$\theta$$ sería decir que el intervalo definido por $$C_N = (a,b)$$ 
hace que $$P_{\theta}(\theta \in C_N) \geq 1 - \alpha$$. Es importante
mencionar que el intervalo no mide la probabilidad de $$\theta$$ dado que 
$$\theta$$ es una constante, en su lugar mide de que el valor estimado
esté dentro de esos límites con esa probabilidad. Por ejemplo, si el parámetro
se estima con el mismo procedimiento, con diferentes muestras, 
100 veces, un intervalo de 95% de confianza dice que 95 veces la estimación 
estará en el intervalo. 

El error estándar (_standard error_) de $$\hat \theta_N$$ es $$se = se(\hat \theta_N) = \sqrt{V(\hat \theta_N)}$$. Si asumimos que 
$$\hat \theta_N = \frac{1}{N}\sum_{i=1}^N X_i$$ donde $$X_i \sim P$$, 
es decir, $$X_i$$ es seleccionada de manera independiente de 
la distribución $$P$$ con media $$\mu$$ y desviación estándar $$\sigma$$. 
Entonces $$E[\hat \theta_N] = E[\frac{1}{N} \sum X_i] = \frac{1}{N} \sum E [X_i] = \frac{1}{N} N \mu = \mu$$. Con respecto a la varianza, se observa que usando la siguiente propiedad $$V(\sum_{i=1}^N a_i X_i) = \sum_{i=1}^N a_i^2V(X_i)$$; se tiene $$V(\hat \theta_N) = V(\sum_{i=1}^N \frac{1}{N} X_i) = \sum \frac{1}{N^2}V(X_i) = \frac{1}{N^2} \sum V(X_i) = \frac{N\sigma^2}{N^2} = \frac{\sigma^2}{N}$$.

El Teorema del Limite Central dice que $$\bar X_N \approx \mathbb N(\mu, \frac{\sigma^2}{N})$$, donde $$\bar X_N = \frac{1}{N} \sum_{i=1}^N X_i$$ y 
corresponde a la suma de $$N$$ variables aleatorias independientes. 
Tomando en cuenta lo anterior es factible asumir que $$\hat \theta_N \approx \mathbb N(\mu, \frac{\sigma^2}{N})$$. En este caso el intervalo se define como:

$$C_N = (\hat \theta_N - z_{\frac{\alpha}{2}}se, \hat \theta_N + z_{\frac{\alpha}{2}}se),$$

donde $$z_{\frac{\alpha}{2}} = \Phi^{-1}(1 - \frac{\alpha}{2})$$ 
y $$\Phi$$ es la función de distribución acumulada de una normal. 

## Ejemplo - Accuracy

Recordado que dado una entrada el clasificador puede acertar la clase a la 
que pertenece esa entrada, entonces el resultado se puede modelar como 
$$1$$ si la respuesta es correcta y $$0$$ de lo contrario. En este caso 
la respuesta es una variable aleatoria con una distribución de 
Bernoulli. Recordando que el parámetro 
$$\hat p_N = \frac{1}{N} \sum_{i=1}^N X_i$$ donde
$$X_i$$ corresponde al resultado del algoritmo en el $$i$$-ésimo ejemplo. 
Recordando que la varianza de una distribución Bernoulli es $$p(1-p)$$ y 
$$se=\sqrt{\frac{p(1-p)}{N}}$$, entonces:

$$C_N = (\hat p_N - z_{\frac{\alpha}{2}}\sqrt{\frac{p(1-p)}{N}}, \hat p_N + z_{\frac{\alpha}{2}}\sqrt{\frac{p(1-p)}{N}}).$$

Suponiendo $$N=100$$ y $$p=0.85$$ el siguiente código calcula el intervalo
usando $$\alpha=0.05$$

```python
from scipy.stats import norm
import numpy as np

alpha = 0.05
z = norm().ppf( 1 - alpha / 2)
p = 0.85
N = 100
Cn = (p - z * np.sqrt(p * (1 - p) / N), p + z * np.sqrt(p * (1 - p) / N))
```

$$C_n = (0.78, 0.92)$$.

En el caso anterior se supuso que se contaba con los resultados de
un algoritmo de clasificación, con el objetivo de completar este ejemplo
a continuación se presenta el análisis con un Naive Bayes en el problema
del Iris. 

Lo primero que se realiza es cargar los datos y algunas librerías que ayudarán
en el proceso.

```python
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, StratifiedKFold

X, y = load_iris(return_X_y=True)
```

Lo que interesa es conocer el rendimiento del clasificador en instancias
no vistas para lo cual una manera es dividir el conjunto en un conjunto
de entrenamiento y otro de prueba. Entrenar en el conjunto de entrenamiento
y probar en el de prueba. Las siguientes instrucciones realizan este 
procedimiento. 

```python
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3)
model = GaussianNB().fit(Xtrain, ytrain)
hy = model.predict(Xtest)
``` 

Se califica la clasificación de cada ejemplo y con ese valor 
se estima $$p$$ y se define el número de ejemplos. El resto del
código es equivalente al visto en el ejemplo anterior. 
 
```python
X = np.where(ytest == hy, 1, 0)
p = X.mean()
N = X.shape[0]
Cn = (p - z * np.sqrt(p * (1 - p) / N), p + z * np.sqrt(p * (1 - p) / N))
```

$$C_N = (0.8953, 1.0158)$$.

Cuando se cuenta con conjuntos de datos pequeños y que además no se ha 
definido un conjunto de prueba, otra manera de obtener las predicciones
del algoritmo de clasificación es mediante el uso de validación cruzada
mediante K-fold. En el siguiente código muestra su uso, el cambio 
solamente es en el procedimiento para obtener las predicciones el cual se 
muestra a continuación. 

```python
kf = StratifiedKFold(n_splits=10, shuffle=True)
hy = np.empty_like(y)
for tr, ts in kf.split(X, y):
    model = GaussianNB().fit(X[tr], y[tr])
    hy[ts] = model.predict(X[ts])
```

El resto del código es equivalente al usado previamente y se lista 
en seguida. 

```python
X = np.where(y == hy, 1, 0)
p = X.mean()
N = X.shape[0]
Cn = (p - z * np.sqrt(p * (1 - p) / N), p + z * np.sqrt(p * (1 - p) / N))
```

$$C_N = (0.9196, 0.9871)$$.


# Bootstrap

Existen ocasiones donde no es sencillo identificar el error estándar
($$se$$) y por lo mismo no se puede calcular el intervalo de
confianza. Por ejemplo, en el caso de que se desee medir el intervalo
de $$recall$$ o cualquier otra medida de rendimiento donde no sea
sencillo calcular el error estándar. 

Bootstrap es un procedimiento que nos permite calcular el error 
estándar e intervalos de confianza. Suponiendo que $$T_N = g(X_1, X_2, \ldots, X_N)$$ sea una estadística de $$N$$ variables aleatorias 
tomadas de una distribución $$F$$, y es de interés conocer $$V_F(T_n)$$. 
Un ejemplo que se ha visto es cuando la estadística es la media, es decir,
$$T_N = \bar T_N= \frac{1}{N} \sum_{i=1}^N X_i$$ y 
$$V_F(\bar T_N) = \frac{\sigma^2}{N}$$. 

En el caso general donde no se puede encontrar una solución analítica
para $$V_F(T_N)$$ es donde viene a ser útil el método de Bootstrap. Este
método simula un punto de la distribución $$T_N$$ mediante la selección
con remplazo de $$N$$ elementos del conjunto $${X_1,X_2,\ldots,X_N}$$, 
es decir, el primer elemento sería 
$$T_{N,1}^* = g(X_1^*,X_2^*,\ldots,X_N^*)$$ donde
$$X_i^*$$ es seleccionado con reemplazo. Suponiendo que este
proceso se realizar $$B$$ veces, entonces:

$$\hat V(T_n) = \frac{1}{B} \sum_{i=1}^B (T_{N,i} - \frac{1}{B} \sum_j^B T_{N,j} )^2.$$

Es más sencillo entender este método mediante un ejemplo. Usando el ejercicio
de $$N=100$$ y $$p=0.85$$ y $$\alpha=0.05$$ descrito previamente, el 
siguiente código primero construye las variables aleatorias de tal
manera que den $$p=0.85$$

```python
from scipy.stats import norm
import numpy as np

alpha = 0.05
z = norm().ppf( 1 - alpha / 2)
X = np.zeros(N)
X[:85] = 1
```

$$X$$ es una arreglo que podrían provenir de la evaluación de un
clasificador usando alguna medida de similitud entre predicción
y valor medido. El siguiente paso es generar seleccionar con 
remplazo y obtener $$g$$, para este ejemplo $$g$$ es la media. 
El resultado se guarda en una lista $$B$$ y se repite el experimento
$$500$$ veces.

```python
B = []
for _ in range(500):
    s = np.random.randint(X.shape[0], size=X.shape[0])
    B.append(X[s].mean())
``` 

El error estándar es:

```python
se = np.sqrt(np.var(B))
``` 

y el intervalo de confianza sería en este caso, asumiendo una 
distribución normal. 

```python
Cn = (p - z * se, p + z * se)
```

$$C_N = (0.7746, 0.9254)$$. Se puede observar que previamente
se había obtenido un intervalo de confianza de: $$(0.78, 0.92)$$.

Continuando con el mismo ejemplo pero ahora analizando
Naive Bayes en el problema del Iris. El primer paso es obtener
evaluar las predicciones que se puede observar en el siguiente
código (previamente descrito.)

```python
X, y = load_iris(return_X_y=True)
kf = StratifiedKFold(n_splits=10, shuffle=True)

hy = np.empty_like(y)
for tr, ts in kf.split(X, y):
    model = GaussianNB().fit(X[tr], y[tr])
    hy[ts] = model.predict(X[ts])

X = np.where(y == hy, 1, 0)
```

Realizando la selección de con remplazo y calculando el intervalo

```python
B = []
for _ in range(500):
    s = np.random.randint(X.shape[0], size=X.shape[0])
    B.append(X[s].mean())
se = np.sqrt(np.var(B))
Cn = (p - z * se, p + z * se)
```

$$C_N=(0.9215, 0.9852)$$. Se puede observar que previamente 
se había obtenido un intervalo de $$(0.9196, 0.9871)$$. 

Existe otra manera de calcular los intervalos de confianza
y es mediante el uso del percentil, utilizando directamente
las estimaciones realizadas a $$g$$ en la selección. El siguiente
código muestra este método usando el ejemplo anterior. 

```python
Cn = (np.percentile(B, alpha * 100), np.percentile(B, (1 - alpha) * 100))
```

$$C_N=(0.9263, 0.9800)$$. 

## Ejemplo - Macro-Recall

Hasta el momento se han usado una medida de rendimiento para la cual
se puede conocer su varianza y esto es el accuracy. Existen problemas
donde esta medida no es recomendada, en el siguiente ejemplo utilizaremos
macro-Recall para medir el rendimiento de Naive Bayes en el problema 
del Iris. El primer paso es realizar las predicciones del algoritmo
usando validación cruzada y hacer la muestra con reemplazo $$B$$. 

```python
from scipy.stats import norm
import numpy as np
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score

alpha = 0.05
z = norm().ppf( 1 - alpha / 2)

X, y = load_iris(return_X_y=True)
kf = StratifiedKFold(n_splits=10, shuffle=True)

hy = np.empty_like(y)
for tr, ts in kf.split(X, y):
    model = GaussianNB().fit(X[tr], y[tr])
    hy[ts] = model.predict(X[ts])

B = []
for _ in range(500):
    s = np.random.randint(X.shape[0], size=X.shape[0])
    _ = recall_score(y[s], hy[s], average="macro")
    B.append(_)
```

En el siguiente paso se asume que macro-Recall se comporta como 
una normal y se calcula el intervalo

```python
p = np.mean(B)
se = np.sqrt(np.var(B))
Cn = (p - z * se, p + z * se)
``` 

$$C_N=(0.9098, 0.9855)$$.

Completando el ejercicio, el intervalo se puede calcular directamente
usando el percentil, como se muestra a continuación.

```python
Cn = (np.percentile(B, alpha * 100), np.percentile(B, (1 - alpha) * 100))
```

$$C_N=(0.9167, 0.9765)$$. 







