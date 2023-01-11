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

## Paquetes usados
{: .no_toc .text-delta }
```python
from scipy.stats import multivariate_normal, norm, kruskal
from sklearn.datasets import load_diabetes, load_breast_cancer, load_iris
from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import recall_score
from sklearn.naive_bayes import GaussianNB
from sklearn import decomposition
from EvoMSA.model import GaussianBayes
import numpy as np
from matplotlib import pylab as plt
import seaborn as sns
sns.set_theme()
```

---

# Introducción

Habiendo descrito problemas de clasificación y regresión, podemos imaginar que 
existen ocasiones donde las variables que describen al problema no contribuyen 
dentro de la solución, o que su aporte está dado por otras componentes dentro de la 
descripción. Esto trae como consecuencia, en el mejor de los caso, que el algoritmo 
tenga un mayor costo computacional o en un caso menos afortunado que el algoritmo 
tenga un rendimiento menor al que se hubiera obtenido seleccionado las variables.
Es pertinente mencionar que el caso contrario correspondiente al incremento del 
número de variables es también un escenario factible y se abordará en otra ocasión.

Existen diferentes maneras para reducir la dimensión de un problema, es decir, 
transformar la representación original $$x \in \mathbb R^d $$ a una 
representación $$\hat x \in \mathbb R^m $$ donde $$m < d$$.
El objetivo es que la nueva representación $$\hat x$$ contenga la información
necesaria para realizar la tarea de clasificación o regresión. También otro
objetivo sería reducir a $$\mathbb R^2$$ o $$\mathbb R^3$$ de tal manera que se 
pueda visualizar el problema. En este último caso el objetivo es que se mantengan 
las características de los datos en $$\mathbb R^d$$ en la reducción.  

Esta descripción inicia con una metodología de selección basada en 
calcular estadísticas de los datos y descartar aquellas que no 
proporcionan información de acuerdo a la estadística. 

# Selección de Variables basadas en Estadísticas
{: #sec:seleccion-var-estadistica }

Se utilizará el [problema sintético](/AprendizajeComputacional/capitulos/02Teoria_Decision/#sec:tres-normales)
de tres clases para describir el algoritmo de selección. Este problema está
definido por tres Distribuciones Gausianas donde se generan tres muestras de 
1000 elementos cada una utilizando el siguiente código. 

```python
p1 = multivariate_normal(mean=[5, 5], cov=[[4, 0], [0, 2]])
p2 = multivariate_normal(mean=[1.5, -1.5], cov=[[2, 1], [1, 3]])
p3 = multivariate_normal(mean=[12.5, -3.5], cov=[[2, 3], [3, 7]])
X_1 = p1.rvs(size=1000)
X_2 = p2.rvs(size=1000)
X_3 = p3.rvs(size=1000)
```

Estas tres distribuciones representan el problema de clasificación 
para tres clases. El siguiente código une las 
tres matrices `X_1`, `X_2` y `X_3` y genera un arreglo `y` que representa
la clase. 


```python
D = np.concatenate((X_1, X_2, X_3), axis=0)
y = np.array(['a'] * X_1.shape[0] + ['b'] * X_2.shape[0] + ['c'] * X_3.shape[0])
```

Por construcción el problema está en $$\mathbb R^2$$ y se sabe que las
dos componentes contribuyen a la solución del mismo, es decir, 
imagine que una de las variables se pierde, con la información 
restante se desarrollaría un algoritmo de clasificación con un rendimiento
mucho menor a aquel que tenga toda la información. 

Continuando con el problema sintético, en está ocasión lo que se realiza es
incluir en el problema una variable que no tiene relación con la clase, para
esto se añade una variable aleatoria con una distribución Gausiana 
con $$\mu=2$$ y $$\sigma=3$$ tal como se muestra en el siguiente código.

```python
N = norm.rvs(loc=2, scale=3, size=3000)
D = np.concatenate((D, np.atleast_2d(N).T), axis=1)
```

El objetivo es encontrar la variable que no está relacionada con la salida. 
Una manera de realizar esto es imaginar que si la media en las diferentes
variables es la misma en todas las clases entonces esa variable no contribuye
a discriminar la clase. En la sección [Estimación de Parámetros](/AprendizajeComputacional/capitulos/03Parametricos/#sec:estimacion-parametros)
se presentó el procedimiento para obtener las medias que 
definen $$\mathbb P(\mathcal X \mid \mathcal Y)$$ para cada clase. 
El siguiente código muestra el procedimiento para calcular las medías 
que son $$\mu_1=[4.9694, 5.0407, 1.9354]^T$$, $$\mu_2=[ 1.5279, -1.4644,  2.0522]^T$$
y $$\mu_3=[12.553 , -3.4208,  2.0132]^T$$.

```python
labels = np.unique(y)
[np.mean(D[y==i], axis=0) for i in labels]
```

Se observa que la media de la tercera variable es aproximadamente igual
para las tres clases, teniendo un valor cercano a $$2$$ tal y como fue 
generada. Entonces lo que se busca es un procedimiento que permita 
identificar que las muestras en cada grupo (clase) hayan sido originadas por la 
misma distribución. Es decir se busca una prueba que indique 
que las primeras dos variables provienen de diferentes distribuciones y 
que la tercera provienen de la misma distribución. Es pertinente comentar que este 
procedimiento no es aplicable para problemas de regresión. 

Si se puede suponer que los datos provienen de una Distribución Gausiana entonces
la prueba a realizar es ANOVA, en caso contrario se puede utilizar su equivalente
método no paramétrico como es la prueba Kruskal-Wallis. Considerando que
de manera general se desconoce la distribución que genera los datos, entonces
se presenta el uso de la segunda prueba.  

La prueba Kruskal-Wallis identifica si un conjunto de muestras independientes
provienen de la misma distribución. La hipótesis nula es que las muestras
provienen de la misma distribución. La función `kruskal` implementa esta 
prueba y recibe tantas muestras como argumentos. En el siguiente código 
ilustra su uso, se observa que se llama a la función `kruskal` para cada 
columna en `D` y se calcula su valor $$p$$. Los valores $$p$$
obtenidos son: $$[0.0, 0.0, 0.5938]$$ lo cual indica que para las primeras
dos variables la hipótesis nula se puede rechazar y por el otro lado la hipótesis
nula es factible para la tercera variable con un valor $$p=0.5938$$

```python
res = [kruskal(*[D[y==l, i] for l in labels]).pvalue
       for i in range(D.shape[1])]
```

En lugar de discriminar aquellas características que no aportan a modelar
los datos, es más común seleccionar las mejores características. Este procedimiento
se puede realizar utilizando los valores $$p$$ de la prueba 
estadística o cualquier otra función que ordene la importancia de 
las características. 

El procedimiento equivalente a la estadística de Kruskal-Wallis en regresión
es calcular la estadística F cuya hipótesis nula es asumir que el coeficiente
obtenido en una regresión lineal entre las variables independientes y la 
dependiente es zero. Esta estadística se encuentra implementada en 
la función `f_regression`. El siguiente código muestra su uso en el 
conjunto de datos de diabetes; el cual tiene $$10$$ variables 
independientes. En la variable `p_values` se tienen los valores $$p$$
se puede observar que el valor $$p$$ correspondiente a la segunda variable
tiene un valor de $$0.3664$$, lo cual hace que esa variable no sea representativa
para el problema que se está resolviendo. 


```python
X, y = load_diabetes(return_X_y=True)
f_statistics, p_values = f_regression(X, y)
```

Un ejemplo que involucra la selección de las variables más representativas
mediante una calificación que ordenan la importancia de las mismas se muestra en el 
siguiente código. Se puede observar que las nueve variables seleccionadas 
son: $$[0, 2, 3, 4, 5, 6, 7, 8, 9]$$ descartando la segunda variable que tiene
el máximo valor $$p.$$ 


```python
sel = SelectKBest(score_func=f_regression, k=9).fit(X, y)
sel.get_support(indices=True)
```

## Ventajas y Limitaciones

Las técnicas vistas hasta este momento requieren de pocos recursos computacionales 
para su cálculo. Además están basadas en estadísticas que permite saber 
cuales son las razones de funcionamiento. Estas fortalezas también son origen 
a sus debilidades, estas técnicas observan en cada paso las variables independientes
de manera aislada y no consideran que estas variables pueden interactuar. Por 
otro lado, la selección es agnóstica del algoritmo de aprendizaje utilizado.

# Conjunto de Validación y Validación Cruzada
{: #sec:validacion-cruzada }

Antes de inicia la descripción de otro algoritmo para la selección de 
características es necesario describir otro conjunto que se utiliza
para optimizar los hiperparámetros del algoritmo de aprendizaje. Previamente
se describieron los conjuntos de [Entrenamiento y Prueba](/AprendizajeComputacional/capitulos/03Parametricos/#sec:conjunto-entre-prueba), 
i.e., $$\mathcal T$$ y $$\mathcal G$$. En particular estos conjuntos 
se definieron utilizando todos los datos $$\mathcal D$$ con lo que se especifica
el problema.

La mayoría de algoritmos de aprendizaje tiene hiperparámetros que pueden ser 
ajustados para optimizar su comportamiento al conjunto de datos que se está 
analizando. Estos hiperparámetros pueden estar dentro del algoritmo o pueden ser 
modificaciones al conjunto de datos para adecuarlos al algoritmo. El segundo
caso es el que se analizará en esta unidad. Es decir, se seleccionarán las
variables que facilitan el aprendizaje. 

Para optimizar los parámetros es necesario medir el rendimiento del algoritmo,
es decir, observar como se comporta el algoritmo en el proceso de predicción. 
La manera trivial sería utilizar el conjunto de prueba $$\mathcal G$$ para
medir el rendimiento. Pero es necesario recordar que este conjunto no debe ser
visto durante el aprendizaje y la optimización de los parámetros es parte de ese
proceso. Si se usara $$\mathcal G$$ entonces dejaría de ser el conjunto de prueba
y se tendría que seleccionar otro conjunto de prueba. 

Entonces para optimizar los parámetros del algoritmo se selecciona del 
conjunto de entrenamiento, i.e., $$\mathcal T$$, el 
**conjunto de validación**, $$\mathcal V$$. Este conjunto tiene la característica
que $$\mathcal T \cap \mathcal V \cap \mathcal G = \emptyset$$ 
y $$\mathcal T \cup \mathcal V \cup \mathcal G = \mathcal D.$$ Una manera de realizar
estos es seleccionar primeramente el conjunto de prueba $$\mathcal G$$ 
y de los datos restantes generar los conjuntos de entrenamiento $$\mathcal T$$
y validación $$\mathcal V.$$

Para ejemplificar esta idea se utiliza el [ejemplo](/AprendizajeComputacional/capitulos/04Rendimiento/#ejemplo)
de Breast Cancer Wisconsin utilizando un Clasificador Bayesiano donde
el hiperparámetro es si se utilizar un Bayesiano Ingenuo o se estima
la matriz de covarianza. 

El primer paso es obtener los datos del problema, lo cual se muestra 
en la siguiente instrucción.

```python
D, y = load_breast_cancer(return_X_y=True)
```

Con los datos $$\mathcal D$$ se genera el conjunto de 
prueba $$\mathcal G$$ y los datos para estimar los parámetros y 
optimizar los hiperparámetros del algoritmo. En la variable `T` se 
tiene los datos para encontrar el algoritmo y en `G` se tiene
el conjunto de prueba.

```python
T, G, y_t, y_g = train_test_split(D, y, test_size=0.2)
```

Los datos de entrenamiento y validación se generan de manera equivalente
tal como se muestra en la siguiente instrucción. El conjunto de 
validación ($$\mathcal V$$) se encuentra en la variable `V` y la
variable dependiente en `y_v.`

```python
T, V, y_t, y_v = train_test_split(T, y_t, test_size=0.3)
```

En este momento ya se tienen todos los elementos para medir el redimiento
de cada hiperparámetro. Empezando por el clasificador con la matriz
de covarianza completa. El recall en ambas clases 
es $$[0.9388, 0.9886].$$ 


```python
gaussian = GaussianBayes().fit(T, y_t)
hy_gaussian = gaussian.predict(V)
recall_score(y_v, hy_gaussian, average=None)
```

La segunda opción es utilizar un clasificador Bayesiano Ingenuo, 
el cual se especifica con el parámetro `naive` tal y como se muestra
en las siguientes instrucciones. El recall en las dos clases
es $$[0.8776, 0.9773].$$  

```python
ingenuo = GaussianBayes(naive=True).fit(T, y_t)
hy_ingenuo = ingenuo.predict(V)
recall_score(y_v, hy_ingenuo, average=None)
```

Comparando el rendimiento de los dos hiperparámetros se observa
que el mejor rendimiento se obtuvo con la matriz de covarianza completa. 
Entonces se procede a entrenar el clasificador para probar 
su rendimiento en $$\mathcal G$$, obteniendo un recall en 
las dos clases de $$[0.7895, 1.]$$ tal y como se muestra en el siguiente código

```python
gaussian = GaussianBayes().fit(np.concatenate((T, V)), np.concatenate((y_t, y_v)))
hy_gaussian = gaussian.predict(G)
recall_score(y_g, hy_gaussian, average=None)
```

## k-Iteraciones de Validación Cruzada

Cuando se cuenta con pocos datos para medir el rendimiento del algoritmo
es común utilizar la técnica de _k-fold cross-validation_ la cual consiste
en partir $$k$$ veces el conjunto de entrenamiento para generar $$k$$
conjuntos de entrenamiento y validación. 

La idea se ilustra con la siguiente 
tabla, donde se asume que los datos son divididos en 5 bloques ($$k=5$$),
cada columna de la tabla ilustra los datos de ese bloque. Si los datos
se dividen en $$k=5$$ bloques, entonces existen $$k$$ iteraciones que 
son representadas por cada renglón de la siguiente tabla, quitando 
el encabezado de la misma. La letra en cada celda identifica el uso 
que se le dará a esos datos en la respectiva iteración, es decir, `T`
representa que se usará como conjunto de entrenamiento y [V](){: .btn .btn-green }
se usa para identificar aquellos datos que se usarán como conjunto
de validación.

La idea es entrenar y probar el rendimiento del algoritmo $$k$$ veces
usando las particiones en cada renglón. Es decir, la primera vez
se usan los datos de la primera columna como el conjunto de validación,
y el resto de columnas, $$[2, 3, 4, 5]$$, como conjunto de entrenamiento
para estimar los parámetros del algoritmo. En la segunda iteración se usan
los datos del segundo renglón donde se observa que los datos en la 
cuarta columna corresponden al conjunto de validación y los datos en las
columnas $$[1, 2, 3, 5]$$ son usados como conjunto de prueba. Las 
iteraciones siguen hasta que todos los datos fueron utilizados en una 
ocasión como conjunto de validación. 


|  1 |  2 |  3 |  4 |  5 |
|----|----|----|----|----|
|[V](){: .btn .btn-green }   |T   |T   |T   |T   |
|T   |T   |T   |[V](){: .btn .btn-green }   |T   |
|T   |T   |[V](){: .btn .btn-green }   |T   |T   |
|T   |T   |T   |T   |[V](){: .btn .btn-green } |
|T   |[V](){: .btn .btn-green }   |T   |T   |T   |

Se utiliza el mismo problema para medir el rendimiento
del hiperparámetro del clasificador Gausiano. Lo primero
es seleccionar el conjunto de prueba ($$\mathcal G$$)
que se realiza con el siguiente código. 

```python
T, G, y_t, y_g = train_test_split(D, y, test_size=0.2)
```

La validación cruzada con k-iteraciones se puede realizar
con la clase `KFold` de la siguiente manera. La primera
linea crear una variable para guardar el rendimiento. En la 
segunda linea se inicializa el procedimiento indicando que los 
datos sean tomados al azar. Después se realiza el ciclo con 
las $$k$$ iteraciones, para cada iteración se genera un índice
`ts` que indica cuales son los datos del conjunto de entrenamiento
y `vs` que corresponde a los datos de validación. Se estiman 
los parámetros usando `ts` tal y como se observa en la cuarta linea.
Habiendo estimado los parámetros se predicen los datos del conjunto
de validación (5 linea), se mide el recall en todas las clases y se 
guarda en la lista `perf.` Al final se calcula la media de 
los $$k$$ rendimientos medidos, teniendo un valor de $$[0.8943, 0.9818].$$

```python
perf = []
kfold = KFold(shuffle=True)
for ts, vs in kfold.split(T):
    gaussian = GaussianBayes().fit(T[ts], y_t[ts])
    hy_gaussian = gaussian.predict(T[vs])
    _ = recall_score(y_t[vs], hy_gaussian, average=None)    
    perf.append(_)
np.mean(perf, axis=0)    
```

Un procedimiento equivalente se realiza para el caso 
del clasificador Bayesiano Ingenuo tal y como se muestra a continuación.
La media del recall en las clases es $$[0.828 , 0.9717].$$ 
Se observa que el clasificador Bayesiano con la matriz de covarianza
tiene un mejor rendimiento en validación que el clasificador 
Bayesiano Ingenuo. El último paso sería calcular el rendimiento 
en el conjunto $$\mathcal G$$ lo cual fue presentado anteriormente.  

```python
perf = []
kfold = KFold(shuffle=True)
for ts, vs in kfold.split(T):
    gaussian = GaussianBayes().fit(T[ts], y_t[ts])
    hy_gaussian = gaussian.predict(T[vs])
    _ = recall_score(y_t[vs], hy_gaussian, average=None)    
    perf.append(_)
np.mean(perf, axis=0)  
```

# Selección hacia Adelante

Un procedimiento que pone en el centro del proceso de selección al algoritmo
de aprendizaje utilizado es **Selección hacia Adelante** y su complemento
que sería **Selección hacia Atrás.** El algoritmo de selección hacia 
adelante es un procedimiento iterativo que selecciona una variable a la 
vez guiada por el rendimiento de esta variable cuando es usada en el algoritmo
de aprendizaje. Al igual que los procedimientos anteriores este no modifica
las características del problema, solamente selecciona las que se consideran 
relevantes.

En selección hacia adelante y hacia atrás se inicia con el conjunto de entrenamiento $$\mathcal T = \{(x_i, y_i)\},$$ 
con una función $$L$$ que mide el rendimiento del algoritmo de 
aprendizaje $$\mathcal H.$$
La idea es ir seleccionando de manera iterativa aquellas variables que generan un 
modelo con mejores capacidades de generalización. Para medir la generalización 
del algoritmo se pueden realizar de diferentes maneras, una es mediante la 
división de $$\mathcal X$$ en dos conjuntos: entrenamiento y validación; y la 
segunda manera corresponde a utilizar $$k$$-iteraciones de validación cruzada.  

Suponga un conjunto $$\pi \subseteq \{1, 2, \ldots, d\}$$ 
de tal manera que $$\mathcal T_\pi$$ solamente cuenta con las variables identificadas en el conjunto $$\pi$$. Utilizando está notación el algoritmo se puede definir de la siguiente manera. Inicialmente $$\pi^0 = \emptyset $$, 
en el siguiente paso $$\pi^{j + 1} \leftarrow \pi^j \cup \textsf{arg max}_{\{i \mid i \in \{1, 2, \ldots, d\}, i \notin \pi^j\}} \mathcal P(\mathcal H, \mathcal T_{\pi \cup i}, L)$$, donde $$\mathcal P$$ representa el rendimiento
del algoritmo $$\mathcal H$$ en el subconjunto $${\pi \cup i}$$ usando la 
función de rendimiento $$L$$. Este proceso continua si $$\mathcal P^{j+1} > \mathcal P^j $$ donde $$\mathcal P^0 = 0$$.

Es importante mencionar que el algoritmo antes descrito es un algoritmo voraz y que el encontrar el óptimo de este problema de optimización no se garantiza con este tipo de algoritmos. Lo que quiere decir es que el algoritmo encontrará un óptimo local.

Ilustrando estos pasos en el conjunto de Breast Cancer Wisconsin
utilizado previamente. El primer paso es medir el rendimiento cuando
solamente una variable interviene en el proceso. Como 
inicialmente $$\pi^0 = \emptyset $$ entonces solo es necesario generar 
un clasificador cuando sola una variable está involucrada.
El rendimiento de cada variable se guarda en la variable `perf`, 
se puede observar que el primer ciclo (línea 3) itera por todas las 
variables en la representación, para cada una se seleccionan 
los datos solo con esa variable `T1 = T[:, np.array([var])]`
después se hacen $$k$$-iteraciones de validación cruzada y finalmente
se guarda el rendimiento. El rendimiento que se está calculando
es macro-recall. 

```python
perf = []
kfold = KFold(shuffle=True)
for var in range(T.shape[1]):
    T1 = T[:, np.array([var])]
    perf_inner = []
    for ts, vs in kfold.split(T1):
        gaussian = GaussianNB().fit(T1[ts], y_t[ts])
        hy_gaussian = gaussian.predict(T1[vs])
        _ = recall_score(y_t[vs], hy_gaussian, average='macro')    
        perf_inner.append(_)
    perf.append(np.mean(perf_inner))
```

<!--
sns.barplot(x=list(range(len(perf))), y=perf)
plt.grid()
plt.xlabel('Variable')
plt.ylabel('Macro-Recall')
plt.savefig('var-forward-sel.png', dpi=300)
-->

La siguiente figura muestra el rendimiento de las 30 variables, 
se observa como una gran parte de las variables proporcionan 
un rendimiento superior al $$0.8$$ y la variable que tiene el mejor
rendimiento es la que corresponde al índice $$27$$ y valor $$0.9057.$$

![Selección hacia Adelante](/AprendizajeComputacional/assets/images/var-forward-sel.png)


El algoritmo de selección hacia atrás y adelante se implementa en 
la clase `SequentialFeatureSelector` y su uso se observa en las siguientes
instrucciones. 

```python
kfolds = list(KFold(shuffle=True).split(T))
scoring = make_scorer(lambda y, hy: recall_score(y, hy, average='macro'))
seq = SequentialFeatureSelector(estimator=GaussianNB(),
                                scoring=scoring,
                                n_features_to_select='auto',
                                cv=kfolds).fit(T, y_t)
```

Al igual que en el algoritmo de `SelectKBest` las variables 
seleccionadas se pueden observar con la función `get_support`.
En este caso las variables seleccionadas 
son: $$[1, 4, 8, 9, 11, 14, 16, 18, 19, 20, 21, 22, 23, 27, 28].$$

El siguiente código utiliza selección hacia atrás en el mismo conjunto de 
datos, dando como resultado la siguiente
selección $$[2, 3, 10, 12, 15, 16, 20, 21, 22, 23, 24, 25, 27, 28, 29].$$
Se puede observar que las variables seleccionadas por los métodos 
son diferentes. Esto es factible porque los algoritmos solo aseguran
llegar a un máximo local y no está garantizado que el máximo local
corresponda al máximo global. 

## Ventajas y Limitaciones

Una de las ventajas de la selección hacia atrás y adelante es que el 
algoritmo termina a lo más cuando se han analizado todas las variables, 
esto eso para un problema en $$\mathbb R^d$$ se analizarán un máximo
de $$d$$ variables. La principal desventaja es que estos algoritmos 
son voraces, es decir, toman la mejor decisión en el momento, lo cual 
tiene como consecuencia que sean capaces de garantizar llegar a un
máximo local y en ocasiones este máximo local no corresponde al máximo
global. Con el fin de complementar esta idea, en un problema $$\mathbb R^d$$
se tiene un espacio de búsqueda de $$2^d - 1$$, es decir, se tiene esa 
cantidad de diferentes configuraciones que se pueden explorar. En 
los algoritmos de vistos se observa un máximo de $$d$$ elementos de
ese total.

# Análisis de Componentes Principales

Los algoritmos de selección hacia atrás y adelante tiene la característica de requerir un conjunto de entrenamiento de aprendizaje supervisado, por lo que no podrían ser utilizados en problemas de aprendizaje no-supervisado. En esta sección se revisará el uso de Análisis de Componentes Principales (Principal Components Analysis - PCA) para la reducción de dimensión. PCA tiene la firma: $$f: \mathbb R^d \rightarrow \mathbb R^m $$ donde $$m < d $$

La idea de PCA es buscar una matriz de 
proyección $$W^T \in \mathbb R^{m \times d}$$ tal que los elementos 
de $$\mathcal D = \{x_i\}$$ sea transformados utilizando $$z = W^T x$$
donde $$z \in \mathbb R^m$$. El objetivo es que la muestra $$z_1$$ tenga la mayor variación posible. Es decir, se quiere observar en la primera característica de los elementos transformados la mayor variación; esto se puede lograr de la siguiente manera.

Si suponemos que $$x \sim \mathcal N_d(\mu, \Sigma)$$ y $$ w \in \mathbb R^d $$ entonces $$w^T x \sim \mathcal N(w^T \mu, w^T \Sigma w)$$ y por lo tanto $$\textsf{Var} (w^T x) = w^T \Sigma w $$.

Utilizando esta información se puede describir el problema como encontrar $$w_1 $$ tal que $$\textsf{Var}(z_1)$$ sea máxima, donde $$\textsf{Var}(z_1) = w_1^T \Sigma w_1 $$. Dado que en este problema de optimización tiene multiples soluciones, se busca además maximizar bajo la restricción de $$\mid\mid w_1 \mid\mid = 1$$. Escribiéndolo como un problema de Lagrange quedaría como: $$\max_{w_1} w_1^T \Sigma w_1 - \alpha (w^T_1 w_1 - 1)$$. Derivando con respecto a $$w_1$$ se tiene que la solución es: $$\Sigma w_i = \alpha w_i $$ donde esto se cumple solo si $$w_1 $$ es un eigenvector de $$\Sigma$$ y $$\alpha$$ es el eigenvalor correspondiente. Para encontrar $$w_2$$ se requiere $$\mid\mid w_2 \mid\mid = 1$$ y que los vectores sean ortogonales, es decir, $$w_2^T w_1 = 0$$. Realizando las operaciones necesarias se encuentra que $$w_2$$ corresponde al segundo eigenvector y así sucesivamente.

## Ejemplo - Visualización

Supongamos que deseamos visualizar los ejemplos del problema del iris. Los ejemplos se encuetran en $$\mathbb R^4$$ entonces para poderlos graficar en $$\mathbb R^2$$ se requiere realizar una transformación como podría ser Análisis de Componentes Principales.

Empezamos por cargar los datos del problema tal y como se muestra en la siguiente instrucción.

```python
D, y = load_iris(return_X_y=True)
```

Habiendo importado los datos el siguiente paso es inicializar la clase de PCA, para esto requerimos especificar el parámetro que indica el número de componentes deseado, dado que el objetivo es representar en $$\mathbb R^2$$ los datos, entonces el ocupamos dos componentes. La primera linea inicializa la clase de PCA, después se hace la proyección en la segunda línea y finalmente se grafican los datos.

```python
pca = decomposition.PCA(n_components=2).fit(D)
Xn = pca.transform(D)
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
