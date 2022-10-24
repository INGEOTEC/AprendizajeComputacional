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

---

# Introducción

En sesiones anteriores hemos visto diferentes técnicas para discriminar entre clases. De manera general hemos visto que si se cuenta con una función discrimitante, $$g(x)$$, se puede seleccinar la clase utilizando la siguiente regla:

$$C(x) = \textsf{argmax}_{j=1}^K g_j(x)$$.

Se puede observar que en el caso de métodos paramétricos la función $$g_j(x)$$ corresponde a utilizar $$P(C \mid X)$$, donde los parametros de las funciones de densidad correspondientes a la probabilidad a priori y la verosimilitud se calculan estiman utilizando los datos de entrenamiento. También se describieron los métodos no paramétricos donde la función de densidad no se conoce y se aproxima con estos métodos. 

Por otro lado en esta unidad revisares aquellos métodos donde la función discriminante no se basa en el uso de la verosimilitud, en su lugar se asume que la frontera entre las clases tiene una forma particular, en este caso lineal, que puede separar los datos correctamente. Es decir, la función discriminte puede tener la siguiente forma:

$$g_j(x \mid w_j, w_{j0}) = w_j \cdot x + w_{j0},$$

En el caso general, $$x \in \mathbb R^d$$,  $$w_j \in \mathbb R^d$$ y $$\cdot$$ representa el producto punto entre dos vectores. 

# Clasificación binaria

En clasificación binaría se puede observar que no es necesario definir dos funciones discriminantes, $$g_1$$ y $$g_2$$, lo cual se puede verificar del siguiente razonamiento:

$$
\begin{eqnarray*}
    g(x) &=& g_1(x) - g_2(x) \\
         &=& (w_1 x + w_{10}) - (w_2 x + w_{20}) \\
         &=& (w_1 + w_2) x + (w_{10} - w_{20}) \\
         &=& w x + w_0
\end{eqnarray*}.$$

En este caso la clase está dada por el signo de $$g(x)$$, es decir, $$x$$ corresponda a la clase positiva si $$g(x)>0$$. Se puede observar que la constante $$w_0$$ está actuando como un umbral, es decir, $$x$$ corresponde a la clase positiva si $$w x > - w_0$$. Utilizando esta notación se observa que el origen se encuentra en el lado positivo del hiperplano si $$w_0 > 0$$, de lo contrario se encuentra del lado negativo.

En la siguiente figura se observa el plano (linea verde) que divide las dos clases, este plano representa los puntos que satisfacen $$g(x)=0$$. 

![Discriminante Lineal](/AprendizajeComputacional/assets/images/discriminante.png)

# Múltiples clases

Una manera de tratar un problema de $$k$$ clases, es convertirlo en
$$k$$ problemas de clasificación binarios, a este procedimiento se le conoce como Uno vs Resto. 
La idea es entrenar $$k$$ clasificadores donde la clase positiva corresponde
a cada una de las clases y la clase de negativa se construye con todas las clases 
que no son la clase positiva en esa iteración. Finalmente, la clase predicha corresponde
al clasificador que tiene el valor máximo en la función discriminante. 

En la siguiente figura ejemplifica el comportamiento de esta técnica
en un problema de tres clases y utilizando un clasificador con discrimitante lineal.
Se puede observar como la clase naranja esta separada por una linea de las clases verde
y azul, de igualmente se puede ver como existe una linea que separa la clase azul de las 
otras clases y la tercera linea separa la clase verde de la clase azul y naranja. 

![Discriminante Lineal](/AprendizajeComputacional/assets/images/clases3.png)

Para tratar de explicar con mayor detalle la técnica de Uno vs Resto, utilizaremos 
el clasificador de Bayes Ingenuo para ejemplificar este escenario. En el siguiente
código se cargan las librearías necesarias, así como los datos del Iris. 

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
X, y = load_iris(return_X_y=True)
```

Recordando que la idea es generar $$k$$ clasificadores, lo cuales en el 
siguiente código se guardarán en la lista $$C$$. El ciclo itera por todas
las clases que hay, en la siguiente linear se genera un arreglo $$yb$$ donde
se pondrá generará el problema binario, donde $$1$$ corresponde a la clase positiva  
y $$0$$ a la clase negativa. En las últimas lineas se entrena al clasificador y se
guarda en la lista $$C$$.

```python
C = []
for cl in np.unique(y):
    yb = np.empty_like(y)
    m = y == cl
    yb[m] = 1
    yb[~m] = 0
    _ = GaussianNB().fit(X, yb)
    C.append(_)
```

Finalmente, el procedimiento para predecir sería preguntar a cada clasificador por la
clase y escoger al valor máximo; dado que se está utilizando un clasificador de Bayes Ingenuo, 
podemos utilizar la probabilidad de la clase positiva para este propósito. 
El siguiente código calcula el valor de la probabilidad para la clase positiva en 
los $$k$$ clasificadores, se usa la transpuesta para obtener una matriz donde cada renglón
sea un ejemplo y se tengan $$k$$ columnas.

```python
hy = np.vstack([c.predict_proba(X)[:, 1] for c in C]).T
```

Finalmente la clase se calcularía con el valor máximo como se observa en el siguiente código.

```python
hy = hy.argmax(axis=1)
```

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

# Máquinas de Soporte Vectorial

Las máquinas de soporte vectorial siguen la idea de encontrar una función discriminante lineal que separe las clase. En este clasificador se asume un problema binario y las clases están representadas por $$-1$$ y $$1$$, es decir, $$y \in \{-1, 1\}$$. Entonces, las máquinas de soporte vectorial tratan de encontrar una función con las siguientes características. 

Sea $$x_i$$ un ejemplo que corresponde a la clase $$1$$ entonces se busca $$w$$ tal que

$$w^T x_i + w_0 \geq +1.$$

En el caso contrario, es decir, $$x_i$$ un ejemplo de la clase $$-1$$, entonces 

$$w^T x_i + w_0 \leq -1.$$

Estas ecuaciones se pueden escribir como 

$$(w^T x_i + w_0) y_i \geq +1,$$

donde $$(x_i, y_i) \in \mathcal X$$. 

La función discriminante es $$ w^T x + w_0 $$ y la distancia que existe entre cualquier punto $$x_i$$ al discriminante está dada por 

$$\frac{(w^T x_i + w_0) y_i }{\mid \mid w \mid \mid}$$

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