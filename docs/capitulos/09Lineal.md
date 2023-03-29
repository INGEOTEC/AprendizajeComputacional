---
layout: default
title: Discriminantes Lineales
nav_order: 10
---

# Discriminantes Lineales
{: .fs-10 .no_toc }

El **objetivo** de la unidad es conocer y aplicar diferentes métodos lineales de discriminación para atacar problemas de clasificación.

## Tabla de Contenido
{: .no_toc .text-delta }

1. TOC
{:toc}

## Paquetes usados
{: .no_toc .text-delta }
```python
from sklearn.svm import LinearSVC
from scipy.stats import multivariate_normal
from matplotlib import pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_theme()
```
---

# Introducción

En unidades anteriores se han visto diferentes técnicas para discriminar entre clases;
en particular se ha descrito el uso de la 
probabilidad $$\mathbb P(\mathcal Y \mid \mathcal X)$$ para encontrar la clase más 
probable. Los parámetros de $$\mathbb P(\mathcal Y \mid \mathcal X)$$ se han 
estimado utilizando métodos [paramétricos](/AprendizajeComputacional/capitulos/03Parametricos) y [no paramétricos](/AprendizajeComputacional/capitulos/07NoParametricos/). En está unidad se describe el uso de funciones discriminantes
para la clasificación y su similitud con el uso 
de $$\mathbb P(\mathcal Y \mid \mathcal X).$$

# Función Discriminante
{: #sec:discriminante }

En la unidad de [Teoría de Decisión Bayesiana](/AprendizajeComputacional/capitulos/02Teoria_Decision/#ejemplos) se describió el uso 
de $$\mathbb P(\mathcal Y \mid \mathcal X)$$ para clasificar, se mencionó
que la clase a la que pertenece $$\mathcal X=x$$ es la de mayor probabilidad,
es decir,  

$$C(x) = \textsf{argmax}_{k=1}^K \mathbb P(\mathcal Y=k \mid \mathcal X=x),$$

donde $$K$$ es el número de clases y $$\mathcal Y=k$$ representa la $$k$$-ésima
clase. Considerando que la [evidencia](/AprendizajeComputacional/capitulos/02Teoria_Decision/#teorema-de-bayes) 
es un factor que normaliza, entonces, $$C(x)$$ se puede definir de la siguiente manera. 

$$C(x) = \textsf{argmax}_{k=1}^K \mathbb P(\mathcal X=x \mid \mathcal Y=k)\mathbb P(\mathcal Y=k).$$

Agrupando la probabilidad a priori y verosimilitud en una función $$g_k,$$ 
es decir, $$g_k(x) = P(\mathcal X=x \mid \mathcal Y=k)\mathbb P(\mathcal Y=k),$$  
hace que $$C(x)$$ se sea:

$$C(x) = \textsf{argmax}_{k=1}^K g_k(x).$$

Observando $$C(x)$$ y olvidando los pasos utilizados para derivarla,
uno se puede imaginar que lo único necesario para generar un clasificador
de $$K$$ clases es definir un conjunto de functions $$g_k$$ que separen las
clases correctamente. En esta unidad se presentan diferentes maneras para 
definir $$g_k$$ con la característica de que todas ellas son lineales,
e.g., $$g_k(\mathbf x) = \mathbf w_k \cdot \mathbf x + w_{k_0}.$$

## Clasificación Binaria

La descripción de discriminantes lineales empieza con el caso particular de
dos clases, i.e., $$K=2$$. En este caso $$C(\mathbf x)$$ es encontrar el máximo
de las dos funciones $$g_1$$ y $$g_2$$. Una manear equivalente sería
definir a $$C(\mathbf x)$$ como 

$$C(\mathbf x) = \textsf{sign}(g_1(\mathbf x) - g_2(\mathbf x)),$$

donde $$\textsf{sign}$$ es la función que regresa el signo, entonces solo 
queda asociar el signo positivo a la clase 1 y el negativo a la clase 2. 
Utilizando esta definición se observa lo siguiente

$$
\begin{eqnarray*}
    g_1(\mathbf x) - g_2(\mathbf x) &=& (\mathbf w_1 \cdot \mathbf x + w_{1_0}) - (\mathbf w_2 \cdot \mathbf x + w_{2_0}) \\
         &=& (\mathbf w_1 + \mathbf w_2) \cdot \mathbf x + (w_{1_0} - w_{2_0}) \\
         &=& \mathbf w \cdot \mathbf x + w_0
\end{eqnarray*},$$

donde se concluye que para el caso binario es necesario definir 
solamente una función discriminante y que los parámetros de esta función 
son $$\mathbf w$$ y $$\mathbf w_0.$$ Otra característica que se ilustra
es que el parámetro $$\mathbf w_0$$ está actuando como un umbral, 
es decir, $$\mathbf x$$ corresponde a la clase positiva 
si $$\mathbf w \cdot \mathbf x > -w_0.$$

En la siguiente figura se observa el plano (linea) que divide las dos clases, este plano representa los puntos que satisfacen $$g(\mathbf x)=0$$. 

![Discriminante Lineal](/AprendizajeComputacional/assets/images/discriminante.png)
<details markdown="block">
  <summary>
    Código de la figura
  </summary>

```python
X_1 = multivariate_normal(mean=[15, 20], cov=[[3, -3], [-3, 8]]).rvs(1000)
X_2 = multivariate_normal(mean=[5, 5], cov=[[4, 0], [0, 2]]).rvs(1000)

T = np.concatenate((X_1, X_2))
y_t = np.array(['P'] * X_1.shape[0] + ['N'] * X_2.shape[0])
linear = LinearSVC(dual=False).fit(T, y_t)
w_1, w_2 = linear.coef_[0]
w_0 = linear.intercept_[0]
g_0 = [dict(x1=x, x2=y, tipo='g(x)=0')
       for x, y in zip(T[:, 0], (-w_0 - w_1 * T[:, 0]) / w_2)]
df = pd.DataFrame(g_0 + \
                  [dict(x1=x, x2=y, clase='P') for x, y in X_1] + \
                  [dict(x1=x, x2=y, clase='N') for x, y in X_2]
                 )
ax = sns.scatterplot(data=df, x='x1', y='x2', hue='clase', legend=True)
sns.lineplot(data=df, x='x1', y='x2', ax=ax, hue='tipo', palette=['k'], legend=True)
ax.axis('equal')
```  
</details>
<!--
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 0.65))
plt.tight_layout()
plt.savefig('discriminante.png', dpi=300)
-->

## Geometría de la Función de Decisión
{: #sec:geometria-funcion-decision }

La función discriminante $$g(\mathbf x) = \mathbf w \cdot \mathbf x + w_0$$
tiene una representación gráfica. Lo primero que se observa es que los 
parámetros $$\mathbf w$$ viven en al mismo espacio que los datos, tal y como se 
puede observar en la siguiente figura. 

![Discriminante Lineal y Pesos](/AprendizajeComputacional/assets/images/discriminante_2.png)
<details markdown="block">
  <summary>
    Código de la figura
  </summary>

```python
_ = pd.DataFrame([dict(x1=w_1, x2=w_2, clase='w')])
df = pd.concat((df, _), axis=0)
ax = sns.scatterplot(data=df, x='x1', y='x2', hue='clase', legend=True)
sns.lineplot(data=df, x='x1', y='x2', ax=ax, hue='tipo', palette=['k'], legend=True)
ax.axis('equal')
```
</details>
<!--
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 0.65))
plt.tight_layout()
plt.savefig('discriminante_2.png', dpi=300)
-->

Siguiendo con la descripción, los parámetros $$\mathbf w$$ y la función $$g(\mathbf x)$$
son ortogonales, tal y como se muestra en la siguiente figura. Analiticamente la 
ortogonalidad se define de la siguiente manera. Sea $$\mathbf x_a$$ y $$\mathbf x_b$$
dos puntos en $$g(\mathbf x)=0$$, es decir, 

$$\begin{eqnarray*}
g(\mathbf x_a) &=& g(\mathbf x_b) \\
\mathbf w \cdot \mathbf x_a + w_0 &=& \mathbf w \cdot \mathbf x_b + w_0\\
\mathbf w \cdot (\mathbf x_a -  \mathbf x_b) &=& 0,
\end{eqnarray*}$$

donde el vector $$\mathbf x_a -  \mathbf x_b$$ es paralelo a $$g(\mathbf x)=0$$,
ortogonal a $$\mathbf w$$ y el sub-espacio generado por $$\mathbf w \cdot (\mathbf x_a -  \mathbf x_b) = 0$$
pasa por el origen. 

![Ortogonalidad del Discriminante Lineal y w](/AprendizajeComputacional/assets/images/discriminante_3.png)
<details markdown="block">
  <summary>
    Código de la figura
  </summary>

```python
w = np.array([w_1, w_2]) / np.linalg.norm([w_1, w_2])
len_0 = w_0 / np.linalg.norm([w_1, w_2])
df = pd.DataFrame(g_0 + \
                  [dict(x1=x, x2=y, clase='P') for x, y in X_1] + \
                  [dict(x1=x, x2=y, clase='N') for x, y in X_2] + \
                  [dict(x1=0, x2=0, tipo='lw'),
                   dict(x1=-w[0]*len_0, x2=-w[1]*len_0, tipo='lw')]
                 )
ax = sns.scatterplot(data=df, x='x1', y='x2', hue='clase', legend=True)
sns.lineplot(data=df, x='x1', y='x2', ax=ax, hue='tipo',
             palette=['k'] + sns.color_palette()[2:],
             legend=True)
ax.axis('equal')
```
</details>
<!--
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 0.65))
plt.tight_layout()
plt.savefig('discriminante_3.png', dpi=300)
-->

En la figura anterior, $$\ell \mathbf w$$ corresponde al 
vector $$\mathbf w$$ multiplicado por un factor $$\ell$$ de tal manera que intersecte 
con $$g(\mathbf x)=0.$$ El factor $$\ell$$ corresponde a la distancia que hay del 
origen a $$g(\mathbf x)=0$$ la cual 
es $$\ell = \frac{w_0}{\mid\mid \mathbf w \mid\mid}.$$ El signo de $$\ell$$ 
indica el lado donde se encuentra el origen con respecto a $$g(\mathbf x)=0$$

La siguiente figura muestra en rojo la linea generada 
por $$\mathbf w \cdot \mathbf x=0$$, la función 
discriminante $$g(\mathbf x)=0$$ (negro), la línea puntuada muestra la distancia
entre ellas, que corresponde a $$\ell$$ y el vector $$\mathbf w$$. Visualmente,
se observa que $$\mathbf w$$ está pegado a la linea roja, pero esto solo 
es un efecto de la resolución y estos elementos no se tocan.  

![g(x)=0 y w x =0 ](/AprendizajeComputacional/assets/images/discriminante_4.png)
<details markdown="block">
  <summary>
    Código de la figura
  </summary>

```python
vec = np.array([2, (-w_0 - w_1 * 2) / w_2]) - np.array([1, (-w_0 - w_1 * 1) / w_2])
x_max = T[:, 0].max()
length = np.linalg.norm(np.array([x_max, (-w_0 - w_1 * x_max) / w_2]) -
                        np.array([-w[0]*len_0, -w[1]*len_0]))
vec_der = length * vec / np.linalg.norm(vec)
x_min = T[:, 0].min()
length = np.linalg.norm(np.array([x_min, (-w_0 - w_1 * x_min) / w_2]) -
                        np.array([-w[0]*len_0, -w[1]*len_0]))
vec_izq = -length * vec / np.linalg.norm(vec)

g = [dict(x1=x, x2=(- w_1 * x) / w_2, tipo='wx=0')
     for x in np.linspace(vec_izq[0], vec_der[0])]
df = pd.DataFrame([dict(x1=x, x2=y, clase='P') for x, y in X_1] + \
                  [dict(x1=x, x2=y, clase='N') for x, y in X_2] +\
                  [dict(x1=w_1, x2=w_2, clase='w')] +\
                  g_0 + g)
ax = sns.scatterplot(data=df, x='x1', y='x2', hue='clase', legend=True)
sns.lineplot(data=df, x='x1', y='x2', ax=ax, hue='tipo',
             palette=['k'] + sns.color_palette()[3:],
             legend=True)
ax.plot([vec_der[0], x_max], [vec_der[1], (-w_0 - w_1 * x_max) / w_2], '--',
        color=sns.color_palette()[4])
ax.axis('equal')
```
</details>
<!--
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 0.65))
plt.tight_layout()
plt.savefig('discriminante_4.png', dpi=300)
-->


## Múltiples Clases

Una manera de tratar un problema de $$K$$ clases, es convertirlo en
$$K$$ problemas de clasificación binarios, a este procedimiento 
se le conoce como _Uno vs Resto_. La idea es entrenar $$K$$ clasificadores 
donde la clase positiva corresponde a cada una de las clases y la clase de 
negativa se construye con todas las clases que no son la clase positiva en esa 
iteración. Finalmente, la clase predicha corresponde al clasificador que tiene 
el valor máximo en la función discriminante. 

En la siguiente figura ejemplifica el comportamiento de esta técnica
en un problema de tres clases y utilizando un clasificador con discrimitante lineal.
En la figura se muestra las tres funciones discriminantes $$g_k(\mathbf x)=0$$,
los parámetros escalados de esas funciones, i.e., $$ \ell_k \mathbf w_k$$ y los
datos. Por ejemplo se observa como la clase $$1$$ mostrada en azul, se separa
de las otras dos clases con la función $$g_1(\mathbf x)=0$$, es decir, 
para $$g_1(\mathbf x)=0$$ la clase positiva es $$1$$ y la clase negativa corresponde
a los elementos que corresponde a las clases $$2$$ y $$3.$$

![Discriminante Lineal](/AprendizajeComputacional/assets/images/clases3.png)

# Máquinas de Soporte Vectorial

Es momento de describir algunos algoritmos para estimar los parámetros $$\mathbf w$$,
empezando por las máquinas de soporte vectorial. En este clasificador se asume un problema binario y las clases están representadas por $$-1$$ y $$1$$, 
es decir, $$y \in \{-1, 1\}$$. Entonces, las máquinas de soporte vectorial tratan de encontrar una función con las siguientes características. 

Sea $$\mathbf x_i$$ un ejemplo que corresponde a la clase $$1$$ 
entonces se busca $$\mathbf w$$ tal que

$$\mathbf w \cdot \mathbf x_i + w_0 \geq +1.$$

En el caso contrario, es decir, $$\mathbf x_i$$ un ejemplo de la 
clase $$-1$$, entonces 

$$\mathbf w \cdot \mathbf x_i + w_0 \leq -1.$$

Estas ecuaciones se pueden escribir como 

$$(\mathbf w \cdot \mathbf x_i + w_0) y_i \geq +1,$$

donde $$(\mathbf x_i, y_i) \in \mathcal D.$$ 

La función discriminante es $$g(\mathbf x) = \mathbf w \cdot \mathbf x + w_0$$ y la distancia que existe entre cualquier punto $$\mathbf x_i$$ 
al discriminante está dada por 

$$\frac{(\mathbf w \cdot \mathbf x_i + w_0) y_i}{\mid \mid \mathbf w \mid \mid}.$$

Entonces, se puede ver que lo que se busca es encontrar $$w, w_0$$ de tal manera que cualquier punto $$x_i$$ esté lo mas alejada posible del discriminante, esto se logra minimizando $$w$$, es decir, resolviendo el siguiente problema de optimización:

$$ \min \frac{1}{2} \mid \mid w \mid \mid $$

sujeto a $$ (w^T x_i + w_0) y_i \geq +1, \forall (x_i, y_i) \in \mathcal X $$.

En el siguiente video se muestra el uso de una Máquina de Soporte Vectorial lineal y
se compara contra Regresión Logística

{%include linear_svm.html %}

## Kernel

Existen problemas donde no es posible encontrar una función lineal que discrimine entre las clases, para estos problemas es común utilizar una transformación de tal manera que en el nuevo espacio el problema sea linealmente separable. 

Existen varias funciones que son utilizadas para este fin, en general cualquier función con la forma $$K(x, y) \rightarrow \mathbb R$$ funcionaría.

La idea es que en este nuevo espacio los coeficientes $$ w $$ están asociados a ejemplos del conjunto de entrenamiento, es decir, la clase de un ejemplo $$x$$, estaría dada por:

$$g(x) = \sum_{x_i \in \mathcal X} w_i K(x, x_i) + w_0,$$

donde $$x$$ corresponde a la clase positiva si $$g(x)$$ es positivo, de lo 
contrario sería clase negativa. 

El siguiente video se describe el uso kernel dentro de una Máquina de Soporte Vectorial.

{%include svm.html %}

# Regresión Logística

Antes de iniciar la descripción de regresión logísticas definimos la función $$\textsf{logit}$$
la cual es:

$$\textsf{logit}(y) = \log \frac{y}{1 - y}$$

donde la inversa del $$\textsf{logit}$$ es la función sigmoide, $$\textsf{sigmoid}(x) = \frac{1}{1 + \exp(-x)}$$, es decir $$\textsf{sigmoid}(\textsf{logit}(y)) = y$$.

Una manera de definir a este clasificador es suponiendo que el problema de clasificación es 
binario y definiendo la función discriminante como $$\textsf{logit}(P(C_1 \mid x))$$; 
expandiendo este término se obtiene:

$$
\begin{eqnarray*}
    \textsf{logit}(P(C_1 \mid x)) &=& \log \frac{P(C_1 \mid x)}{1 - P(C_1 \mid x)} \\
                                  &=& \log \frac{P(C_1 \mid x)}{P(C_2 \mid x)} \\
                                  &=& \log \frac{\frac{P(x \mid C_1) P(C_1)}{P(x)}}{\frac{P(x \mid C_2) P(C_2)}{P(x)}} \\
                                  &=&  \log \frac{P(x \mid C_1) P(C_1)}{P(x \mid C_2) P(C_2)} \\
                                  &=& \log \frac{P(x \mid C_1)}{P(x \mid C_2)} + \log \frac{P(C_1)}{P(C_2)}
\end{eqnarray*}
$$

se observar que se usa el hecho de que $$P(C_1 \mid x) + P(C_2 \mid x) = 1$$ y 
el Teorema de Bayes. En métodos paramétricos y no paramétricos se vieron técnicas
para definir la verosimilitud de la clase, i.e., $$P(x \mid C)$$; por el contrario, en 
regresión logística se asume una forma, es decir, se asume:

$$\log \frac{P(x \mid C_1)}{P(x \mid C_2)} = w \cdot x + w_0^v.$$

Tomando en cuenta que $$\log \frac{P(C_1)}{P(C_2)}$$ es una constante, $$w_0^p$$, 
entonces quedaría como: $$ \textsf{logit}(P(C_1 \mid x)) = w \cdot x + w_0$$, 
donde $$w_0 = w_0^v + w_0^p$$. Despejando $$P(C_1 \mid x)$$:

$$ \hat y(x) = P(C_1 \mid x) = \textsf{sigmoid}(w \cdot x + w_0) = \frac{1}{1 + \exp{-(w \cdot x + w_0)}}.$$

Se puede asumir que $$ y \mid x $$ sigue una distribución Bernoulli en el caso de dos clases, entonces la máxima verosimilitud quedaría como:

$$ l(w, w_0 \mid \mathcal X) = \prod_{(x, y) \in \mathcal X} (\hat y(x))^{y} (1 -  \hat y(x)))^{1-y}. $$

Siempre que se tiene que obtener el máximo de una función esta se puede transformar a un 
problema de minimización, por ejemplo, para el caso anterior definiendo como $$E = -\log l$$, 
utilizando esta transformación el problema sería minimizar la siguiente función:

$$E(w, w_0 \mid \mathcal X) = - \sum_{(x, y) \in \mathcal X} y \log \hat y(x) + (1-y) \log (1 -  \hat y(x)).$$

Es importante notar que la ecuación anterior corresponde a Entropía cruzada,
definida como: $$H(P, Q) = - \sum_w P(w) \log Q(w)$$. La Entropía cruzada 
para dos clases, $$x$$ y $$w$$, es:

$$
\begin{eqnarray*}
    H(P, Q) &=& -[P(x) \log Q(x) + P(w) \log Q(w)] \\
            &=& -[P(x) \log Q(x) + (1 - P(x)) \log (1 - Q(x))]
\end{eqnarray*}
$$

utilizando la notación usada en $$E(w, w_0 \mid \mathcal X)$$ se puede observar que 
$$y=P(x)$$ y $$\hat y(x)=Q(x)$$, es decir $$E(w, w_0 \mid \mathcal X) = \sum_{(x, y) \in \mathcal X} H(y, \hat y(x))$$.

Otra característica de $$E(w, w_0 \mid \mathcal X)$$ es que no tiene una solución cerrada y 
por lo tanto es necesario utilizar un método de optimización iterativo para 
encontrar los parámetros $$w$$ y $$w_0$$. 

En el siguiente video se describe el uso de regresión logística 

{%include regresion_logistica.html %}

# Discriminación por Regresión

En regresión se supone que la respuesta del sistema se modela como: $$y = \hat y(x) + \epsilon$$,
donde $$\epsilon \sim \mathcal N(0, \sigma^2)$$. Asumiento que $$y \in \{0, 1\}$$ una forma
de restringir a que los valores de $$\hat y$$ se encuentren en el intervalo adecuado es 
haciendo $$\hat y(x) = \textsf{sigmoid}(w x + w_0)$$.

En estas condiciones la verosimilitud es:

$$ l(w, w_0 \mid \mathcal X) = \prod_{(x, y) \in \mathcal X} \frac{1}{\sqrt{2\pi \sigma}} \exp[\frac{(y - \hat y(x))^2}{2 \sigma^2}]. $$

Al igual que en regresión logística, un problema de maximiación se puede plantear como un
problema de minimización; para el caso anterior quedaría como: 

$$ E(w, w_0 \mid \mathcal X) = \frac{1}{2} \sum_{(x, y) \in \mathcal X} (y - \hat y(x))^2.$$ 

También en este caso, no existe una solución cerrada para encontrar los parámetros $$w$$ y $$w_0$$. 

