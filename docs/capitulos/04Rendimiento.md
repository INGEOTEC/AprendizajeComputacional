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

## Paquetes usados
{: .no_toc .text-delta }
```python
from EvoMSA.model import GaussianBayes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np
```
---

# Introducción

Es importante conocer el rendimiento del algoritmo de aprendizaje
computacional desarrollado. En aprendizaje supervisado la medición
se hace mediante el conjunto de prueba, $$\mathcal G$$, mientras 
que en aprendizaje no supervisado es posible utilizar el conjunto 
de entrenamiento $$\mathcal T$$ o utilizar un conjunto de prueba. 
Es importante notar que aunque en 
el proceso de entrenamiento puede usar una función de rendimiento
para estimar o encontrar el algoritmo que modela los datos,
es importante complementar esta medición con otras
funciones de rendimiento. Esta unidad describe algunas de las medidas
más utilizadas para medir el rendimiento de algoritmos de 
clasificación y regresión. 

# Clasificación

En clasificación existen diferentes medidas de rendimiento, algunas de ellas son accuracy, precision, recall, y $$F_1$$, entre otras. 
En [esta publicación](http://nmis.isti.cnr.it/sebastiani/Publications/ICTIR2015.pdf) se describe de manera axiomática algunas de estas medidas
y se dan recomendaciones en general sobre medidas de rendimiento
para clasificadores. 

Varias de las medidas de rendimiento toman como insume la 
**Tabla de Confusión**, la cual contiene la información del 
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

## Error
{: #sec:error }

Se empieza la descripción con el 
[error de clasificación](/AprendizajeComputacional/capitulos/02Teoria_Decision/#sec:error-clasificacion) el cual es la proporción
de errores y se puede definir como 

$$\textsf{error}(\mathcal Y, \mathcal{\hat Y}) = 1 -  \textsf{accuracy}(\mathcal Y, \mathcal{\hat Y}).$$

## Accuracy
{: #sec:accuracy}

El error se define mediante el accuracy. El accuracy es la proporción
de ejemplos correctamente clasificados, utilizando la notación de la 
tabla de confusión quedaría como:

$$\textsf{accuracy}(\mathcal Y, \mathcal{\hat Y}) = \mathbb P(\mathcal Y=p, \mathcal{\hat Y}=p) + \mathbb P(\mathcal Y=n, \mathcal{\hat Y}=n).$$

Una manera equivalente de ver el accuracy es utilizando la probabilidad
condicional,
es decir, $$\textsf{accuracy}(\mathcal Y, \mathcal{\hat Y}) = \mathbb P( \mathcal{\hat Y}=p \mid \mathcal Y=p)\mathbb P(\mathcal Y=p) + \mathbb P(\mathcal{\hat Y}=n \mid \mathcal Y=n)\mathbb P(\mathcal Y=n).$$ 
Esta manera ayuda a entender el caso cuando se tiene una clase con muchos
ejemplos, e.g., $$\mathbb P(\mathcal Y=p) \gg \mathbb P(\mathcal Y=n),$$
en ese caso se ve que el accuracy está dominado por el primer 
término, i.e., $$\mathbb P( \mathcal{\hat Y}=p \mid \mathcal Y=p)\mathbb P(\mathcal Y=p).$$ En este caso, 
la manera trivial de optimizar el accuracy es crear
un clasificador que siempre regrese la clase $$p.$$ Por esta razón el 
accuracy no es una medida adecuada cuando las clases
son desbalanciadas, es buena medida 
cuando $$\mathbb P(\mathcal Y=p) \approx \mathbb P(\mathcal Y=n).$$

## Precision
{: #sec:precision }

La siguiente medida de rendimiento es la precision, se puede observar en 
la probabilidad condicional es que se conoce la predicciones positivas y de
esas predicciones se mide si son correctas. Basándose en esto, se puede 
ver que una manera de generar un algoritmo competitivo en esta media corresponde
a predecir la clase solo cuando exista una gran seguridad de la clase. 

La segunda ecuación ayuda a medir en base de la tabla de confusión.

$$\begin{eqnarray}
\textsf{precision}_p(\mathcal Y, \mathcal{\hat Y}) &=& \mathbb P(\mathcal Y=p \mid \mathcal{\hat Y}=p)\\
&=& \frac{\mathbb P(\mathcal Y=p, \mathcal{\hat Y}=p)}{\mathbb P(\mathcal{\hat Y}=p)}
\end{eqnarray}$$

## Recall
{: #sec:recall }

El recall complementa la precision, al calcular la probabilidad de 
ejemplos correctamente clasificados como $$p$$ dados todos los ejemplos
que se tienen de la clase $$p$$. En base a esta ecuación se puede observar
que un algoritmo trivial con el máximo valor de recall solamente tiene 
que predecir como clase $$p$$ todos los elementos. 

$$\begin{eqnarray}
\textsf{recall}_p(\mathcal Y, \mathcal{\hat Y}) &=& \mathbb P(\mathcal{\hat Y}=p \mid \mathcal{Y}=p) \\
&=& \frac{\mathbb P(\mathcal{\hat Y}=p, \mathcal{Y}=p)}{\mathbb P(\mathcal Y=p)}
\end{eqnarray}$$

## $$F_\beta$$
{: #sec:f1 }

Finalmente, una manera de combinar el recall con la precision es la 
medida $$F_\beta$$, es probable que esta medida se reconozca más cuando $$\beta=1$$. 
La idea de $$\beta$$ es ponderar el peso que se le quiere dar a la precision
con respecto al recall.

$$F^p_\beta(\mathcal Y, \mathcal{\hat Y}) = (1 + \beta^2) \frac{\textsf{precision}_p(\mathcal Y, \mathcal{\hat Y}) \cdot \textsf{recall}_p(\mathcal Y, \mathcal{\hat Y})}{\beta^2 \cdot \textsf{precision}_p(\mathcal Y, \mathcal{\hat Y}) + \textsf{recall}_p(\mathcal Y, \mathcal{\hat Y})}$$

## Medidas Macro
{: #sec:macro }

En las definiciones de precision, recall y $$F_\beta$$ se ha usado un subíndice
y superíndice con la letra $$p$$ esto es para indicar que la medida se está realizando
con respecto a la clase $$p$$. Esto ayuda también a ilustrar que en un 
problema de $$K$$ clases se tendrán $$K$$ diferentes medidas de precision, 
recall y $$F_\beta;$$ cada una de esas medidas corresponde a cada clase. 

En ocasiones es importante tener solamente una medida que englobe el rendimiento
en el caso de los tres rendimientos que se han mencionado, se puede calcular
su versión macro que es la media de la medida. Esto es para un problema 
de $$K$$ clases la precision, recall y $$F_\beta$$ se definen de la siguiente manera.

$$\textsf{macro-precision}(\mathcal Y, \mathcal{\hat Y}) =  \frac{1}{K}\sum_{k} \textsf{precision}_k(\mathcal Y, \mathcal{\hat Y}),$$ 

$$\textsf{macro-recall}(\mathcal Y, \mathcal{\hat Y}) =  \frac{1}{K}\sum_{k} \textsf{recall}_k(\mathcal Y, \mathcal{\hat Y}),$$ 

$$\textsf{macro-}F_\beta(\mathcal Y, \mathcal{\hat Y}) =  \frac{1}{K}\sum_{k} F^k_\beta(\mathcal Y, \mathcal{\hat Y}).$$ 

## Entropía Cruzada
{: #sec:entropia-cruzada }


Una función de costo que ha sido muy utilizada en redes neuronales y en particular en aprendizaje profundo es la Entropía cruzada (Cross entropy) que para una distribución discreta se define como: $$H(P, Q) = - \sum_x P(x) \log Q(x)$$.

## Ejemplo

El ejemplo de 
[Breast Cancer Wisconsin](/AprendizajeComputacional/capitulos/03Parametricos/#ejemplo-breast-cancer-wisconsin)
se utiliza para ilustrar el uso de la medidas de rendimiento presentadas hasta
el momento. 

```python
X, y = datasets.load_breast_cancer(return_X_y=True)
T, G, y_t, y_g = train_test_split(X, y, test_size=0.2)
gaussian = GaussianBayes().fit(T, y_t)
naive = GaussianBayes(naive=True).fit(T, y_t)
hy_gaussian = gaussian.predict(G)
hy_naive = naive.predict(G)
```


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