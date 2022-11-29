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

Es importante conocer el rendimiento del algoritmo de aprendizaje
computacional desarrollado. En aprendizaje supervisado la medición
se hace mediante el conjunto de prueba, $$\mathcal G$$, mientras 
que en aprendizaje no supervisado es posible utilizar el conjunto 
de entrenamiento $$\mathcal T$$ o utilizar un conjunto de prueba. 

Por otro lado, en el proceso de estimar los parámetros o encontrar
la función $$f$$ que modela los datos, se está optimizando una 
función, en el caso de aprendizaje supervisado esta función 
es una 
[función de error](/AprendizajeComputacional/capitulos/01Tipos/#sec:definiciones-aprendizaje-supervisado) $$L.$$
Por ejemplo, en [regresión](/AprendizajeComputacional/capitulos/03Parametricos/#sec:regresion-ols) $$L$$ es el error al cuadrado.

Aunque en el proceso de entrenamiento se usa una función de error,
es importante medir el rendimiento del algoritmo desarrollado
en otras medidas. Esta sección describe algunas de las medidas
más utilizadas para medir el rendimiento de algoritmos de 
clasificación y regresión. 

<!--
Recordando que en aprendizaje supervisado se cuenta con el conjunto de entrenamiento, $$\mathcal X = \{ (x_1, y_1), \ldots, (x_N, y_N )\}$$, utilizado para encontrar una función $$h^*$$ que se comporta similar a la función generadora de los datos esto mediante la minimización del error empírico $$E(h \mid \mathcal X) = \sum_{(x, y) \in \mathcal X} L(y, h(x))$$.

Por otro lado, con el objetivo de medir la generalidad del algoritmo se cuenta con un conjunto de prueba $$\mathcal T={(x_i, y_i)}$$ para $$i=1 \ldots M$$ donde $$\mathcal X \cap \mathcal T = \emptyset$$. En $$\mathcal T$$ también se puede medir error empírico o cualquier otra medida de rendimiento.
-->

# Clasificación

En clasificación existen diferentes medidas de rendimiento, algunas de ellas son accuracy, precision, recall, y $F_1$, entre otras. 
En [esta publicación](http://nmis.isti.cnr.it/sebastiani/Publications/ICTIR2015.pdf) se describe de manera axiomática algunas de estas medidas
y se dan recomendaciones en general sobre medidas de rendimiento
para clasificadores. 

Varias de las medidas de rendimiento toman como insume la 
Tabla de confusión, la cual contiene la información del 
proceso de clasificación. La siguiente tabla muestra la estructura
de esta tabla para un problema binario, donde se tiene una clase
positiva identificada con $$p$$ y una clase negativa ($$n$$).
La variable $$\mathcal Y$$ indica las clases reales y la
variable $$\mathcal{\hat Y}$$ representa la estimación (predicción)
hecha por el clasificador. Adicionalmente, la tabla se puede 
extender a $$K$$ clases siguiendo la misma 
estructura; la diagonal contienen los elementos correctamente 
identificados y los elementos fuera de la diagonal muestra
los errores. 

|                |$$\mathcal{\hat Y}=p$$|$$\mathcal{\hat Y}=n$$|
|----------------|----------------------|----------------------|
|$$\mathcal Y=p$$|Verdaderos Pos.       |Falsos Neg.           |
|$$\mathcal Y=n$$|Falsos Pos.           |Verdaderos Neg.       |

La tabla se puede ver como valores nominales, es decir contar el número
de ejemplos clasificados como verdaderos positivos o como 
proporción de tal manera que las cuatro celdas sumen $$1$$. En esta
descripción se asume que son proporcionen, esto porque se
seguirá una interpretación probabilística descrita
en [este artículo](https://link.springer.com/chapter/10.1007/978-3-540-31865-1_25) para presentar las diferentes medidas de rendimiento.


$$\textsf{accuracy}(\mathcal Y, \mathcal{\hat Y}) = \mathbb P(\mathcal Y=p, \mathcal{\hat Y}=p) + \mathbb P(\mathcal Y=n, \mathcal{\hat Y}=n)$$

$$\begin{eqnarray}
\textsf{precision}_p(\mathcal Y, \mathcal{\hat Y}) &=& \mathbb P(\mathcal Y=p \mid \mathcal{\hat Y}=p)\\
&=& \frac{\mathbb P(\mathcal Y=p, \mathcal{\hat Y}=p)}{\mathbb P(\mathcal{\hat Y}=p)}
\end{eqnarray}$$


$$\begin{eqnarray}
\textsf{recall}_p(\mathcal Y, \mathcal{\hat Y}) &=& \mathbb P(\mathcal{\hat Y}=p \mid \mathcal{Y}=p) \\
&=& \frac{\mathbb P(\mathcal{\hat Y}=p, \mathcal{Y}=p)}{\mathbb P(\mathcal Y=p)}
\end{eqnarray}$$


$$F^+_\beta(\mathcal Y, \mathcal{\hat Y}) = (1 + \beta^2) \frac{\textsf{precision}_p(\mathcal Y, \mathcal{\hat Y}) \cdot \textsf{recall}_p(\mathcal Y, \mathcal{\hat Y})}{\textsf{precision}_p(\mathcal Y, \mathcal{\hat Y}) + \textsf{recall}_p(\mathcal Y, \mathcal{\hat Y})}$$


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