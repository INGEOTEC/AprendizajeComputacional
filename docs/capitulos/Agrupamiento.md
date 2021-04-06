---
layout: default
title: Agrupamiento
nav_order: 7
---

# Agrupamiento
{: .fs-10 .no_toc }

El **objetivo** de la unidad es conocer y aplicar k-medias


## Tabla de Contenido
{: .no_toc .text-delta }

1. TOC
{:toc}

---

# Introducción

En aprendizaje no supervisado se cuenta con un conjunto \(\mathcal X = \{ x_i \} \), en particular en agrupamiento lo que intersa es asociar a cada elemento \(x_i\) a un grupo en particular, es decir, \(x_i \in G_j \) donde \(\cup_j^K G_j = \mathcal X \) y \(G_j \cap G_i = \emptyset\) para todo \(i \neq j \).

Para encontrar está partición se ocupa algún criterio de optimización, en particular este criterio de optimización en K-medias es el reducir la distancia entre un la media del grupo y los elementos del mismo.

# K-medias

De manera formal lo que se K-medias es encontrar la partición \(G = \{G_1, G_2, \ldots, G_K \} \) tal que \(\textsf{arg min}_G \sum_{i=1}^K \sum_{x \in G_i} \mid\mid x - \mu_i \mid\mid \), donde \(\mu_i\) corresponde a la media de todos los elementos que pertenecen a \(G\).

La forma en que este algoritmo inicia es generando aleatoriamente k-\(\mu\) y asociando a cada elemento \(x\) a la media con el mínimo error. De esta manera se tienen todos los elementos para calcular \(\sum_{i=1}^K \sum_{x \in G_i} \mid\mid x - \mu_i \mid\mid \) y utilizando está información podemos derivar con respecto a \(\mu_i\) la ecuación anterior para encontrar la regla para actualizar \(mu_i\).

Realizando lo anterior se obtiene que \(\mu_i = \frac{1}{\mid G_i \mid}\sum_{x \in G_i} x \) para el siguiente paso. El proceso continua hasta que los grupos no cambian o se llega al número máximo de iteraciones.

## Ejemplo

En el siguiente ejemplo se usará K-medias para encontrar 2 y 3 grupos en el conjunto del iris.

El primer paso es cargar los datos y las libreías que utilizaremos, para poder visualizar los resultados se utilizará PCA para hacer una transformación de los datos.

```python
from sklearn import datasets
from matplotlib import pylab as plt
from sklearn import decomposition
from sklearn.cluster import KMeans
X, y = datasets.load_iris(return_X_y=True)
``` 

La clase se inicializa primero con 2 grupos (primera linea), se predice los grupos para todo el conjunto y la siguientes lineas solo son para visualizar los grupos generados en \(\mathbb R^2\).

```python
m = KMeans(n_clusters=2).fit(X)
cl = m.predict(X)
# visualización
pca = decomposition.PCA(n_components=2).fit(X)
Xn = pca.transform(X)
for kl in np.unique(cl):
    m_ = cl == kl
    plt.plot(Xn[m_, 0], Xn[m_, 1], '.')
``` 

El resultado se puede observar en la siguiente figura

![K-medias dos grupos](/AprendizajeComputacional/assets/images/kmeans-2grp.png)


Recordando que K-medias se tiene k vectores pivotes, \(\mu_i\) lo cuales son el centroide del grupo, entonces uno podría preguntarse donde se encuentran esos elementos. El siguiente código representa estos pivotes con una estrella en el plano junto con los otros elementos del conjunto de entrenamiento que se visualizaron en la figura anterior.

```python
pca = decomposition.PCA(n_components=2).fit(X)
Xn = pca.transform(X)
for kl in np.unique(cl):
    m_ = cl == kl
    plt.plot(Xn[m_, 0], Xn[m_, 1], '.')
# transformar lo pivotes al plano
C = pca.transform(m.cluster_centers_)
plt.plot(C[:, 0], C[:, 1], '*', color="black")
``` 

![Centros](/AprendizajeComputacional/assets/images/kmeans-2grp-c.png)


Recordando que en aprendizaje no supervisado la clases no se encuentran disponible y son estimadas por el algoritmo, dado que el problema analizado es un problema de clasificación, podemos usar K-medias para estimar las grupos y con este estimado medir el accuracy con las clases reales. Realizando el procedimiento anterior podemos observar que el accuracy es de: 0.893.