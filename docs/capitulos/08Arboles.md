---
layout: default
title: Árboles de Decisión
nav_order: 9
---

# Árboles de Decisión
{: .fs-10 .no_toc }

El **objetivo** de la unidad es conocer y aplicar árboles de decisión a problemas de clasificación y 
regresión.

## Tabla de Contenido
{: .no_toc .text-delta }

1. TOC
{:toc}

## Paquetes usados
{: .no_toc .text-delta }
```python
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.inspection import DecisionBoundaryDisplay
from scipy.stats import multivariate_normal
from matplotlib import pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_theme()
```
---

# Introducción

Los árboles de decisión son una estructura de datos jerárquica, la cual se 
construye utilizando una estrategia de divide y vencerás. Los árboles son un 
método no paramétrico diseñado para problemas
de regresión y clasificación. 

El árbol se camina desde la raíz hacia las hojas; en cada nodo se tiene una 
regla que muestra el camino de acuerdo a la entrada y la hoja indica la clase 
o respuesta que corresponde a la entrada.

# Clasificación
{: #sec:clasificacion }

Utilizando el procedimiento para generar [tres Distribuciones Gausianas](/AprendizajeComputacional/capitulos/02Teoria_Decision/#sec:tres-normales)
se generan las siguientes poblaciones con 
medias $$\mu_1=[5, 5]^T$$, $$\mu_2=[-5, -10]^T$$ y $$\mu_3=[15, -6]^T$$; 
utilizando las matrices de covarianza originales.

![Tres Distribuciones Gausianas](/AprendizajeComputacional/assets/images/clases3-arboles.png)
<details markdown="block">
  <summary>
    Código de la figura
  </summary>

```python
X_1 = multivariate_normal(mean=[5, 5], cov=[[4, 0], [0, 2]]).rvs(1000)
X_2 = multivariate_normal(mean=[-5, -10], cov=[[2, 1], [1, 3]]).rvs(1000)
X_3 = multivariate_normal(mean=[15, -6], cov=[[2, 3], [3, 7]]).rvs(1000)
df = pd.DataFrame([dict(x=x, y=y, clase=1) for x, y in X_1] + \
                  [dict(x=x, y=y, clase=2) for x, y in X_2] + \
                  [dict(x=x, y=y, clase=3) for x, y in X_3])
sns.relplot(data=df, kind='scatter',
            x='x', y='y', hue='clase')
```
</details>
<!--
plt.savefig('clases3-arboles.png', dpi=300)
-->

Con estas tres poblaciones, donde cada distribución genera una clase
se crea un árbol de decisión. El árbol se muestra en la siguiente
figura, donde se observa, en cada nodo interno, la siguiente información. La 
primera linea muestra el identificador del nodo, la segunda corresponde
a la función de corte, la tercera línea es la entropía 
($$H(\mathcal Y) = -\sum_{y \in \mathcal Y} \mathbb P(\mathcal Y=y) \log_2 \mathbb P(\mathcal Y=y)$$), 
la cuarta es el número de elementos que llegaron al nodo y 
la última la frecuencia de cada clase en ese nodo. Por ejemplo, 
la raíz (#0) tiene la función de corte $$x \leq 10.517$$, tiene una entropía 
de $$1.585$$, recibió $$3000$$ elementos y cada clase tiene $$1000$$ ejemplos.

Los hojas (nodos #2, #3, #5, y #6) no cuentan con una función de corte, 
dado que son la parte final del árbol. En el árbol mostrado se observa
que la entropía en todos los casos es $$0$$, lo cual indica que todos los 
elementos que llegaron a ese nodo son de la misma clase. No en todos los 
casos las hojas tienen entropía cero y existen parámetros en la creación
del árbol que permiten crear árboles más simples. Por ejemplo, la hoja #6
tiene solamente un ejemplo, uno se podría preguntar ¿qué pasaría si esa
hoja se elimina? El resultado es tener un árbol más simple. 

![Árbol](/AprendizajeComputacional/assets/images/tree.png)
<details markdown="block">
  <summary>
    Código de la figura
  </summary>

```python
X = np.concatenate((X_1, X_2, X_3), axis=0)
y = np.array([1] * 1000 + [2] * 1000 + [3] * 1000)
arbol = tree.DecisionTreeClassifier(criterion='entropy').fit(X, y)
_ = tree.plot_tree(arbol, node_ids=True,
                   feature_names=['x', 'y'], label='root')
```
</details>
<!--
plt.savefig('tree.png', dpi=300)
-->

La siguiente figura muestra el árbol generado cuando el nodo #6 se quita. Se observa
un árbol con menos nodos, aunque la entropía es diferente de cero en la hoja #4. 
La segunda parte de la figura muestra la función de decisión que genera el árbol 
de decisión. Se observa que cada regla divide el espacio en dos. La raíz (#0)
divide los datos utilizando $$x \leq 10.517$$, donde todos los elementos para los cuales
la función es verdadera se envían al nodo izquierdo (#1), de lo contrario se envían a la 
hoja derecha (#4). Al nodo #1 llegan $$1999$$ los cuales se divide en
utilizando $$y \leq -1.812$$, dando como resultado $$1000$$ en su hoja izquierda (#2)
y $$999$$ en su hoja derecha (#3). 



![Árbol de Decisión y su Función de Decisión](/AprendizajeComputacional/assets/images/tree-funcion-decision.png)
<details markdown="block">
  <summary>
    Código de la figura
  </summary>

```python
ax = plt.subplot(2, 1, 1)
arbol = tree.DecisionTreeClassifier(criterion='entropy', min_samples_split=1002).fit(X, y)
_ = tree.plot_tree(arbol, node_ids=True,
                   feature_names=['x', 'y'], label='root')
ax = plt.subplot(2, 1, 2)
DecisionBoundaryDisplay.from_estimator(arbol, X, cmap=plt.cm.RdYlBu,
                                       response_method='predict',
                                       ax=ax, xlabel='x', ylabel='y')
for i, color in enumerate('ryb'):
    mask = y == (i + 1)
    plt.scatter(X[mask, 0], X[mask, 1], c=color,
        label=f'{i+1}', cmap=plt.cm.RdYlBu, edgecolor='black')
```
</details>
<!--
plt.savefig('tree-funcion-decision.png', dpi=300)
-->

Utilizando el árbol mostrado en la figura anterior, se puede explicar el proceso
de clasificar un nuevo elemento. Por ejemplo, el 
elemento $$\mathbf u=(x=-3, y=0.5)$$ (que se muestra en la siguiente 
figura como un punto negro) se clasifica de la siguiente manera. La función de 
corte ($$x \leq 10.517$$) de la raíz (#0) indica $$\mathbf u$$ pasa al nodo izquierdo #1.
La función de corte ($$y \leq -1.812$$) del nodo #1 indica que $$\mathbf u$$ se
envía a la hoja #3. La clase mayoritaria ($$999$$ elementos) en la hoja #3 
es la clase $$1$$, entonces $$\mathbf u$$ pertenece a la clase $$1$$ de acuerdo 
al árbol de decisión generado. 

![Predicción](/AprendizajeComputacional/assets/images/tree-prediccion.png)
<details markdown="block">
  <summary>
    Código de la figura
  </summary>

```python
DecisionBoundaryDisplay.from_estimator(arbol, X, cmap=plt.cm.RdYlBu,
                                       response_method='predict',
                                       xlabel='x', ylabel='y')
for i, color in enumerate('ryb'):
    mask = y == (i + 1)
    plt.scatter(X[mask, 0], X[mask, 1], c=color,
                label=f'{i+1}', cmap=plt.cm.RdYlBu,
                edgecolor='black')

plt.scatter([-3], [0.5], c='k',
            label=f'{i+1}', cmap=plt.cm.RdYlBu, 
            edgecolor='black')
```
</details>
<!--
plt.savefig('tree-prediccion.png', dpi=300)
-->

## Entrenamiento
{: #sec:clasificacion-entrenamiento }

La construcción un árbol se realiza mediante un procedimiento recursivo en donde 
se aplica la función de corte $$f_m(\mathbf x) = x_i \leq a$$ en el nodo $$m$$, donde 
el parámetro $$a$$ y la componente $$x_i$$ se identifican utilizando los datos
que llegan al nodo $$m$$ de tal manera que se maximice una función de costo.

Una función de costo podría estar basada en la entropía, es decir, para cada 
posible corte se mide la entropía en los nodos generados y se calcula la esperanza
de la entropía de la siguiente manera. 

$$L(x_i, a) = \sum_h \frac{\mid \mathcal D_h \mid}{\mid \mathcal D_m \mid}  H(\mathcal D_h),$$

donde $$H(\mathcal D_h)$$ es la entropía de las etiquetas del conjunto $$\mathcal D_h$$,
la entropía se puede calcular con la siguiente función. La función recibe un 
arreglo con las clases, está protegida para calcular $$0 \log 0 = 0$$ y 
finalmente regresa la entropía de `arr`.

```python
def H(arr):
    a, b = np.unique(arr, return_counts=True)
    b = b / b.sum()
    return - (b * np.log2(b, where=b != 0)).sum()
```

La función que optimiza $$L(x_i, a),$$ para encontrar $$a$$ 
se implementa en el procedimiento `corte_var`. Este procedimiento 
asume que las etiquetas (`labels`) están ordenadas por la variable $$x_i$$, es decir
la primera etiqueta corresponde al valor mínimo de $$x_i$$ y la última al valor 
máximo. Considerando esto, el valor de $$a$$ es el índice con el menor costo.
En la primera linea se inicializa la variable `mejor` para guardar el valor
de $$a$$ con mejor costo. La segunda linea corresponde a $$\mid \mathcal D_m \mid$$,
en la tercera línea se identifican los diferentes valores de $$a$$ que se tiene
que probar, solo se tienen que probar aquellos puntos donde cuando la clase cambia 
con respecto al elemento adyacente, esto se calcula con la función `np.diff`; 
dado que está quita el primer elemento entonces es necesario incrementar $$1.$$
El ciclo es por todos los puntos de corte, se calculan el costo para 
los elementos que están a la izquierda y derecha del corte y se compara el resultado
con el costo con menor valor encontrado hasta el momento. La última línea regresa
el costo mejor así como el índice donde se encontró. 

```python
def corte_var(labels):
    mejor = (np.inf, None)
    D_m = labels.shape[0]
    corte = np.where(np.diff(labels))[0] + 1
    for j in corte:
        izq = labels[:j]
        der = labels[j:]
        a = (izq.shape[0] / D_m) * H(izq)
        b = (der.shape[0] / D_m) * H(der)
        perf = a + b
        if perf < mejor[0]:
          mejor = (perf, j)
    return mejor
```

En el siguiente ejemplo se usa la función `corte_var`; la función regresa
un costo de $$0.459$$ y el punto de corte es el elemento $$3$$, se puede observar
que es el mejor punto de corte en el arreglo dado. 

```python
corte_var(np.array([0, 0, 1, 0, 0, 0]))
```

Con la función `corte_var` se optimiza el valor $$a$$ de $$L(x_i, a)$$, ahora
es el turno de optimizar $$x_i$$ con respecto a la función de costo. El procedimiento
`corte` encuentra el mínimo con respecto de $$x_i$$, está función recibe los índices
(`idx`) donde se buscará estos valores, en un inicio `idx` es un arreglo de $$0$$
al número de elemento del conjunto $$\mathcal D$$ menos uno. La primera línea 
define la variable donde se guarda el menor costo, en la segunda línea se ordenan 
las variables, la tercera línea se obtienen las etiquetas involucradas. El ciclo
va por todas las variables $$x_i$$. Dentro del ciclo se llama a la función 
`corte_var` donde se observa como las etiquetas van ordenadas de acuerdo a la variable
que se está analizando; la función regresa el corte con menor costo y se compara
con el menor costo obtenido hasta el momento, si es menor se guarda en `mejor`. 
Finalmente, se regresa `mejor` y los índices ordenados para poder identificar
los elementos del hijo izquierdo y derecho.  

```python
def corte(idx):
    mejor = (np.inf, None, None)
    orden = np.argsort(X[idx], axis=0)
    labels = y[idx]
    for i, x in enumerate(orden.T):
        comp = corte_var(labels[x])
        if comp[0] < mejor[0]:
            mejor = (comp[0], i, comp[1])
    return mejor, idx[orden[:, mejor[1]]]
```    

Con la función `corte` se puede encontrar los parámetros de la función de
corte $$f_m(\mathbf x) = x_i \leq a$$ para cada nodo del árbol completo 
del ejemplo anterior. Por ejemplo, la función para la raíz (#0) que se observa
en la figura es $$f_{\#0}(\mathbf x) = x \leq 10.517$$, el siguiente código
siguiente se utiliza para encontrar estos parámetros. 

```python
best, orden = corte(np.arange(X.shape[0]))
perf, i, j = best
(X[orden[j], i] + X[orden[j-1], i]) / 2
```

La variable `orden` tiene la información para dividir el conjunto dado,
lo cual se realiza en las siguientes instrucciones, donde `idx_i` corresponde
a los elementos a la izquierda y `idx_d` son los de la derecha. 

```python
idx_i = orden[:j]
idx_d = orden[j:]
```

Teniendo los elementos a la izquierda y derecha, se puede calcular los parámetros
de la función de corte del nodo #1 que son $$f_{\#1}(\mathbf x) = y \leq -1.812$$,
esto se puede verificar con el siguiente código.

```python
best, orden = corte(idx_i)
perf, i, j = best
(X[orden[j], i] + X[orden[j-1], i]) / 2
```

Equivalentemente, para $$f_{\#4}(\mathbf x) = y \leq 3.488$$ se tiene 

```python
best, orden = corte(idx_d)
perf, i, j = best
(X[orden[j], i] + X[orden[j-1], i]) / 2
```

## Ejemplo: Breast Cancer Wisconsin

Se utiliza el conjunto de datos de Breast Cancer Wisconsin para ejemplificar
el algoritmo de Árboles de Decisión. Las siguientes instrucciones se
descargan los datos y se dividen en los conjuntos de entrenamiento
y prueba.  

```python
X, y = load_breast_cancer(return_X_y=True)
T, G, y_t, y_g = train_test_split(X, y, test_size=0.2)
```

La siguiente instrucción entrena un árbol de decisión utilizando 
como función de costo la entropía. En la librería se encuentran 
implementadas otras funciones como el coeficiente Gini y 
[Entropía Cruzada](/AprendizajeComputacional/capitulos/04Rendimiento/#sec:entropia-cruzada) (_Log-loss_). 

```python
arbol = tree.DecisionTreeClassifier(criterion='entropy').fit(T, y_t)
```

Como es de esperarse la predicción se realiza con el método `predict`
como se ve a continuación. 

```python
hy = arbol.predict(G)
```

El error en el conjunto de prueba $$\mathcal G$$ es $$0.0351$$,
se puede comparar este error con otros algoritmos utilizados
en este conjunto como clasificadores paramétricos basados en 
distribuciones [Gausianas](/AprendizajeComputacional/capitulos/03Parametricos/#sec:gaussina-perf-breast_cancer). La siguiente instrucción
muestra el cálculo del error. 

```python
(y_g != hy).mean()
```

Un dato interesante, considerando los parámetros con los que se inicializó
el árbol, entonces este hizo que todas las hojas fueran puras, es decir,
con entropía cero. Por lo tanto el error de clasificación en el conjunto 
de entrenamiento $$\mathcal T$$ es cero, como se puede verificar con
el siguiente código. 

```python
(y_t != arbol.predict(T)).mean()
```


# Regresión

Hasta este momento se ha visto como se optimizan los parámetros de la función de corte $$f_m$$ para 
problemas de clasificación. 

La única diferencia con problemas de regresión es la función que se utiliza para optimizar los 
parámetros, en el caso de clasificación es entropía y en el caso de regresión podría ser el error 
cuadrático medio o la suma de los errores absolutos. 
