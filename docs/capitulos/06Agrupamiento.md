---
layout: default
title: Agrupamiento
nav_order: 7
---

# Agrupamiento
{: .fs-10 .no_toc }

El **objetivo** de la unidad es conocer y aplicar el algoritmo de 
agrupamiento k-medias


## Tabla de Contenido
{: .no_toc .text-delta }

1. TOC
{:toc}

## Paquetes usados
{: .no_toc .text-delta }
```python
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn import decomposition
from sklearn import metrics
from scipy import linalg
import numpy as np
from pandas as pd
from matplotlib import pylab as plt
import seaborn as sns
sns.set_theme()
```

---

# Introducción

Esta unidad trata el problema de agrupamiento, el cual es un 
problema de [aprendizaje no supervisado](/AprendizajeComputacional/capitulos/01Tipos/#sec:aprendizaje-no-supervisado), en 
el cual se cuenta con un 
conjunto $$\mathcal D = \{ x_i \mid i=1, \ldots, N\}$$ donde $$x_i \in \mathbb R^d.$$
El objetivo de agrupamiento es separar los elementos de $$\mathcal D$$
en $$K$$ grupos. Es decir asociar 
a $$x_i \in \mathcal D$$ a un grupo $$x_i \in G_j$$ 
donde $$\cup_j^K G_j = \mathcal D$$
y $$G_j \cap G_i = \emptyset$$ para todo $$i \neq j.$$

Por supuesto existen diferentes algoritmos que se han desarrollado para
generar esta partición, en particular, todos de ellos encuentran la participación
optimizando una función objetivo que se considera adecuada para el problema 
que se está analizando. En particular, esta unidad se enfoca a describir
uno de los algoritmos de agrupamiento más utilizados que es K-medias. 

# K-medias

De manera formal el objetivo de K-medias es encontrar la 
partición $$G = \{G_1, G_2, \ldots, G_K \}$$ que corresponda  
al $$\min \sum_{i=1}^K \sum_{x \in G_i} \mid\mid x - \mu_i \mid\mid,$$ 
donde $$\mu_i$$ es la media de todos los elementos que pertenecen 
a $$G_i.$$

Para comprender la función objetivo ($$\min \sum_{i=1}^K \sum_{x \in G_i} \mid\mid x - \mu_i \mid\mid$$) de k-medias se explican los dos componentes principales 
que son las medias $$\mu_i$$ y los grupos $$G_i.$$

Para ilustrar tanto a $$\mu_i$$ como a $$G_i$$ se utiliza el problema 
del [iris](/AprendizajeComputacional/capitulos/05ReduccionDim/#sec:visualizacion-iris)
cuyos datos se pueden obtener de la siguiente manera.

```python
D, y = load_iris(return_X_y=True)
```

## $$\mu_i$$

Como se describió, $$\mu_i$$ es la media de los elementos que corresponden
al grupo $$G_i$$. Asumiendo que el grupo $$1$$ ($$G_1$$) tiene $$10$$ elementos
seleccionados de manera aleatoria de $$\mathcal D$$ como se muestra a 
continuación.

```python
index = np.arange(len(D))
np.random.shuffle(index)
sel = index[:10]
G_1 = D[sel]
```

La variable `G_1` tiene los 10 elementos considerados como miembros de $$G_1$$
entonces $$\mu_1$$ se calcula como la media de cada componente, lo cual se 
puede calcular con el siguiente código. 

```python
mu_1 = G_1.mean(axis=0)
```

<!--
pca = decomposition.PCA(n_components=2).fit(D)
Xn = pca.transform(G_1)
mu = pca.transform(np.atleast_2d(G_1.mean(axis=0)))[0]
data = pd.DataFrame(dict(x=Xn[:, 0], y=Xn[:, 1], tipo=['G_1'] * Xn.shape[0]))
data.loc[Xn.shape[0]] = dict(x=mu[0], y=mu[1], tipo='mu_1')
sns.scatterplot(data, hue='tipo', x='x', y='y')
plt.savefig('iris-k-means-g1.png', dpi=300)
-->

La siguiente figura muestra los elementos seleccionados ($$x \in G_1$$) 
y la media ($$\mu_1$$) del grupo. Los elementos se encuentran
en $$\mathbb R^4$$ y para visualizarlos se transformaron usando PCA
descrito [previamente.](/AprendizajeComputacional/capitulos/05ReduccionDim/#sec:visualizacion-iris)


![Grupo uno en primera iteración](/AprendizajeComputacional/assets/images/iris-k-means-g1.png)


## $$G_i$$

El complemento del procedimiento anterior es encontrar los elementos
de $$G_i$$ dando la $$\mu_i$$. El ejemplo consiste en generar dos
medias, es decir, $$K=2$$ y encontrar los elementos que corresponden 
a las medias generadas. Se puede utilizar cualquier procedimiento 
para generar dos vectores de manera aleatoria, pero en este ejemplo
se asume que estos vectores corresponden a dos elementos de $$\mathcal D.$$
Estos elementos son los que se encuentran en los indices $$50$$ y $$100$$
tal y como se muestra en las siguientes instrucciones. 

```python
mu_1 = D[50]
mu_2 = D[100]
```

El elemento $$x$$ pertenece al grupo $$G_i$$ si el 
valor $$\mid\mid x - \mu_i\mid\mid$$ corresponde al $$\min_j \mid\mid x - \mu_j\mid\mid.$$
Entonces se requiere calcular $$\mid\mid x - \mu_i\mid\mid$$ para cada 
elemento $$x \in \mathcal D$$ y para cada una de las medias $$\mu_i$$.
Esto se puede realizar con la siguiente instrucción

```python
dis = np.array([linalg.norm((D - np.atleast_2d(mu)), axis=1)
                for mu in [mu_1, mu_2]]).T
```

donde se puede observar que el ciclo itera por cada una de las 
medias, i.e., `mu_1` y `mu_2`. Después se calcula la norma
utilizando la función `linalg.norm` y finalmente se regresa
la transpuesta para tener una matriz de 150 renglones y dos
columnas que corresponden al número de ejemplos en $$\mathcal D$$ y a las dos medias. 
Los valores de `dis[50]` y `dis[100]` son `[0., 1.8439]` y `[1.8439, 0.]` respectivamente.
Tal y como se espera porque $$\mu_1$$ corresponde al índice 50 y $$\mu_2$$
es el índice 100. Estos dos ejemplos, `[0., 1.8439]` y `[1.8439, 0.]`, permiten
observar que el argumento mínimo de `dis` identifica al grupo del elemento,
haciendo la consideración que el índice 0 representa $$G_1$$ 
y el índice 1 es $$G_2.$$ La siguiente instrucción muestra como 
se realiza esta asignación. 

```python
G = dis.argmin(axis=1)
```

<!--
pca = decomposition.PCA(n_components=2).fit(D)
D_pca = pca.transform(D)
Xn = pca.transform(D)
G_1 = Xn[np.where(G == 0)]
G_2 = Xn[np.where(G)]
G_1 = pd.DataFrame(dict(x=G_1[:, 0], y=G_1[:, 1], tipo=['G_1'] * G_1.shape[0]))
G_2 = pd.DataFrame(dict(x=G_2[:, 0], y=G_2[:, 1], tipo=['G_2'] * G_2.shape[0]))
mu = np.vstack((D_pca[50], D_pca[100]))
mu_data = pd.DataFrame(dict(x=mu[:, 0], 
                            y=mu[:, 1],
                            tipo=['mu'] * mu.shape[0]))
data = pd.concat((G_1, G_2, mu_data))
sns.scatterplot(data, hue='tipo', x='x', y='y')
plt.savefig('iris-k-means-g.png', dpi=300)
-->

La siguiente figura muestra los grupos formados, el primer grupo `G_1`
se encuentra en azul y el segundo en naranja, también muestra los elementos
que fueron usados como medias de cada grupo; estos elementos
se observan en color verde. 

![Grupos en primera iteración](/AprendizajeComputacional/assets/images/iris-k-means-g.png)


## Algoritmo

Habiendo explicado $$\mu_i$$ y $$G_i$$ se procede a describir
el procedimiento para calcular los grupos utilizado por k-medias.
Este es un procedimiento iterativo que consta de los siguientes pasos. 

1. Se generar $$K$$ medias de manera aleatoria, donde $$\mu_i$$ corresponde a $$G_i$$
2. Para cada media, $$\mu_i$$, se seleccionan los elementos más cercanos, esto es, $$x \in G_i$$ si $$\mid\mid x - \mu_i\mid\mid$$ corresponde al $$\min_j \mid\mid x - \mu_j\mid\mid.$$
3. Se actualizan las $$\mu_i$$ con los elementos de $$G_i$$
4. Se regresa al paso 2. 

El procedimiento termina cuando se llega a un número máximo de iteraciones o 
que la variación de los $$\mu_i$$ es mínima, es decir, que los grupos
no cambian. 


# Ejemplo: Iris

En el siguiente ejemplo se usará K-medias para encontrar 2 y 3 grupos 
en el conjunto del iris. La clase se inicializa primero con 2 grupos 
(primera linea). En la segunda instrucción se predice los grupos 
para todo el conjunto de datos. Las medias para cada grupo 
se encuentran en el atributo `cluster_centers_.`

```python
m = KMeans(n_clusters=2).fit(D)
cl = m.predict(D)
```

<!--
pca = decomposition.PCA(n_components=2).fit(D)
D_pca = pca.transform(D)
mu = pca.transform(m.cluster_centers_)
mu_data = pd.DataFrame(dict(x=mu[:, 0],
                            y=mu[:, 1],
                            tipo=['mu'] * mu.shape[0]))
data = pd.DataFrame(dict(x=D_pca[:, 0],
                         y=D_pca[:, 1],
                         tipo=[f'G_{x+1}' for x in cl]))
data = pd.concat((data, mu_data))
sns.scatterplot(data, hue='tipo', x='x', y='y')
plt.savefig('kmeans-2grp.png', dpi=300)
-->

La siguiente figura muestra el resultado del algoritmo k-means
en el conjunto del Iris, se muestran los dos grupos $$G_1$$ y 
$$G_2$$ y en color verde $$\mu_1$$ y $$\mu_2$$. 

![K-medias dos grupos](/AprendizajeComputacional/assets/images/kmeans-2grp.png)


Un procedimiento equivalente se puede realizar para generar 
tres grupos, el único cambio es el parámetro `n_clusters`
en la clase `KMeans` de la siguiente manera.

```python
m = KMeans(n_clusters=3).fit(D)
cl = m.predict(D)
```

<!--
pca = decomposition.PCA(n_components=2).fit(D)
D_pca = pca.transform(D)
mu = pca.transform(m.cluster_centers_)
mu_data = pd.DataFrame(dict(x=mu[:, 0],
                            y=mu[:, 1],
                            tipo=['mu'] * mu.shape[0]))
data = pd.DataFrame(dict(x=D_pca[:, 0],
                         y=D_pca[:, 1],
                         tipo=[f'G_{x+1}' for x in cl]))
data = pd.concat((data, mu_data))
sns.scatterplot(data, hue='tipo', x='x', y='y')
plt.savefig('kmeans-3grp.png', dpi=300)
-->

La siguiente figura muestra los tres grupos y con sus 
tres respectivas medias en color rojo. 

![K-medias tres grupos](/AprendizajeComputacional/assets/images/kmeans-3grp.png)

Recordando que en aprendizaje no supervisado no se tiene una variable
dependiente que predecir. En este caso particular se utilizó 
un problema de clasificación para ilustrar el procedimiento de k-medias,
entonces se cuenta con una clase para cada elemento $$x \in \mathcal D$$. 
Además se sabe que el problema del iris tiene tres clases, 
entonces utilizando los tres grupos obtenidos 
previamente podemos medir que tanto 
se parecen estos tres grupos a las clases del iris. Es decir,
se puede saber si el algoritmo de k-medias agrupa los elementos 
de tal manera que cada grupo corresponda a una clase del iris. 

Los grupos generados se encuentran en la lista `cl` y las 
clases se encuentran en `y`. La lista `y` tiene organizada las clases
de la siguiente manera: los primeros 50 elementos son la clase $$0$$, los 
siguientes $$50$$ son clase $$1$$ y los últimos son la clase $$2$$.
Dado que K-medias no conoce los clases y genera los grupos empezando 
de manera aleatoria, entonces es probable que los grupos sigan
una numeración diferente al problema del iris. Los grupos en `cl` 
están organizados de la siguiente manera aproximadamente los $$50$$
primeros elementos son del grupo $$1$$, los siguientes son grupo $$0$$
y finalmente los últimos son grupo $$2$$. Entonces se puede 
renombrar los grupos $$1$$ como $$0$$ y $$0$$ como $$1$$ para que 
correspondan a las clases del iris. 

```python
cambio = np.array([1, 0, 2])
cl_cambio = cambio[cl]
```

Utilizando `cl_cambio` se calcula 
el [accuracy](/AprendizajeComputacional/capitulos/04Rendimiento/#sec:accuracy)
utilizando la siguiente instrucción. Se obtiene un accuracy 
de $$0.8933;$$ que significa que el 89% de los datos
se agrupan en un conjunto que corresponde a la clase del
conjunto del iris. 

```python
metrics.accuracy_score(y, cl_cambio)
```
