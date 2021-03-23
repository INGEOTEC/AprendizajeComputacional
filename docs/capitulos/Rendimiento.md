---
layout: default
title: Medidas de Rendimiento
nav_order: 5
---

# Medidas de Rendimiento
{: .fs-10 .no_toc }

El **objetivo** es conocer las características de diferentes medidas de rendimiento en aprendizaje supervisado así como su uso.

## Tabla de Contenido
{: .no_toc .text-delta }

1. TOC
{:toc}

---

# Introducción

Recordando que en aprendizaje supervisado se cuenta con el conjunto de entrenamiento, $$\mathcal X = \{ (x_1, y_1), \ldots, (x_N, y_N )\}$$, utilizado para encontrar una función $$h^*$$ que se comporta similar a la función generadora de los datos esto mediante la minimización del error empírico $$E(h \mid \mathcal X) = \sum_{(x, y) \in \mathcal X} L(y, h(x))$$.

Por otro lado, con el objetivo de medir la generalidad del algoritmo se cuenta con un conjunto de prueba $$\mathcal T={(x_i, y_i)}$$ para $$i=1 \ldots M$$ donde $$\mathcal X \cap \mathcal T = \emptyset$$. En $$\mathcal T$$ también se puede medir error empírico o cualquier otra medida de rendimiento.

# Clasificación

En clasificación existen diferentes medidas de rendimiento, algunas de ellas son accuracy, precision, recall, y F1 score, entre otras. 

En [http://nmis.isti.cnr.it/sebastiani/Publications/ICTIR2015.pdf]() se describe de manera axiomática algunas de estas medidas y en el siguiente video se verá de manera práctica. 

{%include rendimiento_clasificacion.html %}

Una función de costo que ha sido muy utilizada en redes neuronales y en particular en aprendizaje profundo es la Entropía cruzada (Cross entropy) que para una distribución discreta se define como: $$H(P, Q) = - \sum_x P(x) \log Q(x)$$.

{%include entropia.html %}

# Regresión

Con respecto a regresión las siguientes funciones son utilizadas como medidas de rendimiento.

Error cuadrático medio (Mean Square Error): $$mse(y, \hat y) = \frac{1}{N} \sum_{i=1}^N (y_i - \hat y_i)^2 $$

Error absoluto medio (Mean Absolute Error): $$mae(y, \hat y) = \frac{1}{N} \sum_{i=1}^N \mid y_i - \hat y_i \mid $$

Mean Absolute Percentaje Error: $$mape(y, \hat y) = \frac{1}{N} \sum_{i=1}^N \mid \frac{y_i - \hat y_i}{y_i}\mid $$

La proporción de la varianza explicada por el modelo: $$R^2(y, \hat y) = 1 - \frac{\sum_{i=1}^N (y_i - \hat y_i)^2)}{\sum_{i=1}^N (y_i - \bar y_i)^2)} $$

# Validación Cruzada

{%include kfold.html %}

Continuando con la descripción de validación cruzada vamos a ver un ejemplo de Stratified K-fold cross-validation en el problema de iris y usando Naive Bayes.

El primer paso seria importar las librerías necesarias y los datos.

```python
from sklearn import datasets
from sklearn import model_selection
from sklearn import naive_bayes
import numpy as np
from sklearn import metrics
X, y = datasets.load_iris(return_X_y=True)
````

Definimos para una $$k$$ de 30 e iniciamos la clase correspondiente

```python
K = 30
kfold = model_selection.StratifiedKFold(shuffle=True, n_splits=K)
P = []
```
Finalmente guardamos en la variable P el accuracy y se calcula la media

```python
for tr, vs in kfold.split(X, y):
    m = naive_bayes.GaussianNB().fit(X[tr], y[tr])
    yh = m.predict(X[vs])
    _ = metrics.accuracy_score(y[vs], yh)
    P.append(_)
print("Media: ", np.mean(P))
```