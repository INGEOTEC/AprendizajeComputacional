---
layout: default
title: Reducción de Dimensión
nav_order: 6
---

# Reducción de Dimensión
{: .fs-10 .no_toc }

El **objetivo** de la unidad es aplicar técnicas de reducción de dimensionalidad, para mejorar el aprendizaje y para visualizar los datos


## Tabla de Contenido
{: .no_toc .text-delta }

1. TOC
{:toc}

---

# Introducción

Habiendo descrito problemas de clasificación y regresión, podemos imaginar que existen ocasiones donde las variables que describen al problema no contribuyen dentro de la solución, o que su aporte está dado por otras componentes dentro de la descripción. Esto trae como consecuencia, en el mejor de los caso, que el algoritmo tenga un mayor costo computacional o en un caso menos afortunado que el algoritmo tenga un rendimiento menor al que se hubiera obtenido seleccionado las variables.

Es pertinente mencionar que el caso contrario correspondiente al incremento del número de variables es también un escenario factible y lo trataremos en otra ocasión.

Existen diferentes maneras para reducir la dimensión de un problema, es decir, transformar la representación original $$x \in \mathbb R^d $$ a una respresentación $$\hat x \in \mathbb R^m $$ donde $$d < m $$. El primer algoritmo que se describe es Selección hacia Adelante (Forward Selection). Este algoritmo corresponde a seleccionar de manera iterativa aquellas variables que para un algoritmo particular de aprendizaje supervisado es importante, donde el caso base es iniciar con ninguna variable. El complemento de Selección hacia adelante es selección hacia atrás (Backward Selection) donde se empieza con todas las características y se va eliminando una por una.

Se puede observar que las variables se encuentran sin ninguna modificación en la nueva representación, pero existe una vertiente de algoritmos de reducción de dimensión transforman las entradas originales, es decir, se genera una función tal que $$f: \mathbb R^d \rightarrow \mathbb R^m $$.

# Selección hacia Adelante

En selección hacia adelante y hacia atrás se inicia con el conjunto de entrenamiento $$\mathcal X = \{(x_i, y_i)\}$$, con una función de error $$L$$ y un algoritmo de aprendizaje. La idea es ir selecciónando de manera iterativa aquellas variables que generan un modelo con mejores capacidades de generalización. Para medir la generalización del algoritmo se pueden realizar de diferentes maneras, una es mediante la división de $$\mathcal X$$ en dos conjuntos: entrenamiento y validación; y la segunda manera corresponde a utilizar un k-Fold Cross-validation.

Supongamos que contamos con una función que mide el rendimiento, $$\mathcal P: \mathcal H \times \mathcal X \times L \rightarrow \mathbb R $$, tal que nos regresa el rendimiento del algoritmo en un conjunto de validación o mediante k-Fold cross-validaton, usando la medida de error $$L$$.

Suponga un conjunto $$\pi \subseteq \{1, 2, \ldots, d\} $$ de tal manera que $$\mathcal X_\pi $$ solamente cuenta con las variables identificadas en el conjunto $$\pi$$. Utilizando está notación el algoritmo se puede definir de la siguiente manera. Inicialmente $$\pi^0 = \emptyset $$, en el siguiente paso $$\pi^{j + 1} \leftarrow \pi^j \cup \textsf{arg min}_{\{i \mid i \in \{1, 2, \ldots, d\}, i \notin \pi^j\}} \mathcal P(\mathcal H, \mathcal X_{\pi \cup i}, L)$$. Este proceso continua si $$\mathcal P^{j+1} < \mathcal P^j $$ donde $$\mathcal P^0 = \infty$$.

Es importante mencionar que el algoritmo antes descrito es un algoritmo voraz y que el encontrar el óptimo de este problema de optimizaón no se garantiza con este tipo de algoritmos. Lo que quiere decir es que el algoritmo encontrará un óptimo local.

## Implementando $$\mathcal P$$

Empezando la implementación de la Selección hacia Adelante con $$\mathcal P$$

```python
def P(H, X, L):
    from sklearn import model_selection
    K = 30
    kfold = model_selection.StratifiedKFold(shuffle=True, n_splits=K)
    X, y = X
    P = []
    for tr, vs in kfold.split(X, y):
        m = H().fit(X[tr], y[tr])
        yh = m.predict(X[vs])
        _ = L(y[vs], yh)
        P.append(_)
    return np.mean(P)
```

Se puede observar el uso de Stratified K-fold cross-validation y se utilizaron las mismas variables que en la definición para facilitar la identificación de los elementos.

El siguiente código se podría utilizar para probar la función anterior. Aunque hay que considerar que accuracy no es una función de error, pero en este momento no es relevante dado que solamente estamos probando el funcionamiento de la función.

```python  
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

X, y = datasets.load_iris(return_X_y=True)
print(P(GaussianNB, (X, y), accuracy_score))
```  

## Implementando - Selección hacia Adelante

Inicializando las variables para generar el proceso de selección, observe que accuracy es transformado a una función de error. Suponemos que la variable $$X$$ ya ha sido definida y es una matriz de $$N \times d$$.

```python
H = GaussianNB
pi = set()
P_j = np.inf
L = lambda y, hy: 1 - accuracy_score(y, hy)
d = X.shape[1]
``` 

El ciclo del algoritmo se puede implementar de la siguiente manera.

```python
for _ in range(d):
    arg_min = [(i, P(H, (X[:, np.array(list(pi.union([i])))], y), L))
               for i in set(range(d)) - pi]
    arg_min.sort(key=lambda x: x[1])
    i, P_jplus = arg_min[0]
    if P_j > P_jplus:
        P_j = P_jplus
        pi.add(i)
        continue
    break
```

Donde las primeras tres líneas, después de la instrucción del ciclo, calculan $$ \textsf{arg min}_{\{i \mid i \in \{1, 2, \ldots, d\}, i \notin \pi^j\}} \mathcal P(\mathcal H, \mathcal X_{\pi \cup i}, L)$$. Las variables seleccionadas se encuentran en la variable pi.

{%include forward_selection.html %}

# Análisis de Componentes Principales

Los algoritmos de selección hacia atrás y adelante tiene la caracterísitca de requerir un conjunto de entrenamiento de aprendizaje supervisado, por lo que no podrían ser utilizados en problemas de aprendizaje no-supervisado. En esta sección se revisará el uso de Análisis de Componentes Principales (Principal Components Analysis - PCA) para la reducción de dimensión. PCA tiene la firma: $$f: \mathbb R^d \rightarrow \mathbb R^m $$ donde $$m < d $$

La idea de PCA es buscar una matriz de proyección $$W^T \in \mathbb R^{m \times d}$$ tal que los elementos de $$\mathcal X = \{x_i\} $$ sea transformados utilizando $$z = W^T x$$ donde $$z \in \mathbb R^m $$. El objetivo es que la muestra z_1 tenga la mayor variación posible. Es decir, se quiere observar en la primera característica de los elementos transformados la mayor variación; esto se puede lograr de la siguiente manera.

Si suponemos que $$x \sim \mathcal N_d(\mu, \Sigma)$$ y $$ w \in \mathbb R^d $$ entonces $$w^T x \sim \mathcal N(w^T \mu, w^T \Sigma w)$$ y por lo tanto $$\textsf{Var} (w^T x) = w^T \Sigma w $$.

Utilizando esta información se puede describir el problema como encontrar $$w_1 $$ tal que $$\textsf{Var}(z_1)$$ sea máxima, donde $$\textsf{Var}(z_1) = w_1^T \Sigma w_1 $$. Dado que en este problema de optimización tiene multiples soluciones, se busca además maximizar bajo la restricción de $$\mid\mid w_1 \mid\mid = 1$$. Escribiéndolo como un problema de Lagrange quedaría como: $$\max_{w_1} w_1^T \Sigma w_1 - \alpha (w^T_1 w_1 - 1)$$. Derivando con respecto a $$w_1$$ se tiene que la solución es: $$\Sigma w_i = \alpha w_i $$ donde esto se cumple solo si $$w_1 $$ es un eigenvector de $$\Sigma$$ y $$\alpha$$ es el eigenvalor correspondiente. Para encontrar $$w_2$$ se requiere $$\mid\mid w_2 \mid\mid = 1$$ y que los vectores sean ortogonales, es decir, $$w_2^T w_1 = 0$$. Realizando las operaciones necesarias se encuentra que $$w_2$$ corresponde al segundo eigenvector y así sucesivamente.

## Ejemplo - Visualización

Supongamos que deseamos visualizar los ejemplos del problema del iris. Los ejemplos se encuetran en $$\mathbb R^4$$ entonces para poderlos graficar en $$\mathbb R^2$$ se requiere realizar una transformación como podría ser Análisis de Componentes Principales.

Empezamos por importar las librerías necesarias así como los datos del problema, tal y como se muestra en las siguientes instrucciones.

```python
from sklearn import datasets
from sklearn import decomposition
import numpy as np
from matplotlib import pylab as plt
X, y = datasets.load_iris(return_X_y=True)
```

Habiendo importado los datos el siguiente paso es inicializar la clase de PCA, para esto requerimos especificar el parámetro que indica el número de componentes deseado, dado que el objetivo es representar en $$\mathbb R^2$$ los datos, entonces el ocupamos dos componentes. La primera linea inicializa la clase de PCA, después se hace la proyección en la segunda línea y finalmente se grafican los datos.

```python
pca = decomposition.PCA(n_components=2).fit(X)
Xn = pca.transform(X)
plt.plot(Xn[:, 0], Xn[:, 1], '.')
``` 

El resultado de este proceso se puede observar en la siguiente figura.

![PCA](/AprendizajeComputacional/assets/images/pca.png)

En este problema nos podría ayudar el identificar cada punto con la clase a la que pertenecen usando un color por cada clase. Esto se puede lograr con el siguiente código.

```python 
for kl in np.unique(y):
    m = y == kl
    plt.plot(Xn[m, 0], Xn[m, 1], '.')
```  

El resultado obtenido es:

![PCA con color](/AprendizajeComputacional/assets/images/pca-color.png)

Se puede observar en este ejercicio como la clase representada por el color azul es linealmente separable de las otras dos clases.
