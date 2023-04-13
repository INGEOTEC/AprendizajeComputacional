---
layout: default
title: Optimización
nav_order: 11
---

# Optimización
{: .fs-10 .no_toc }

El **objetivo** de la unidad es conocer y aplicar el método de **Descenso de Gradiente** y **Propagación hacia Atrás** par estimar los parámetros de modelos de clasificación y regresión.

## Tabla de Contenido
{: .no_toc .text-delta }

1. TOC
{:toc}

---

# Introducción

Existen diferentes modelos de clasificación y regresión donde no es posible
encontrar una solución analítica para estimar los parámetros, por ejemplo en 
Regresión Logística. Es en este escenario donde se voltea a métodos de 
optimización iterativos para calcular los parámetros. 

En esta unidad se describe posiblemente el método de optimización más conocido es **Descenso de Gradiente**. Este método como su nombre lo indica utiliza
el gradiente como su ingrediente principal; se describirá como se puede 
calcular el gradiente utilizando un método gráfico y como este método 
naturalmente realiza **Propagación hacia Atrás**.

# Descenso por Gradiente

En un modelo de clasificación y regresión interesa encontrar un vector de 
parámetros, $$w^*$$, que minimicen una función de error, $$E$$, de la siguiente 
manera:

$$w^* = \textsf{argmin}_w E(w \mid \mathcal X)$$

En el caso de que $$E(w \mid \mathcal X)$$ sea una función diferenciable, 
el gradiente está dado por:

$$\nabla_w E(w \mid \mathcal X) = [\frac{\partial E}{\partial w_1}, \frac{\partial E}{\partial w_2}, \ldots]^\intercal.$$

La idea general es tomar la dirección opuesta al gradiente para encontrar el 
mínimo de la función. Entonces el cambio de parámetro está dado por 

$$
\begin{eqnarray*}
\Delta w &=& - \eta \nabla_w E\\
       w &=& w + \Delta w \\
       &=& w - \eta \nabla_w E
\end{eqnarray*}
$$

## Ejemplo - Regresión Lineal

Suponiendo que se quieren estimar los parámetros de la siguiente ecuación 
lineal: $$ f(x) = a x - b$$, para lo cual se tiene un conjunto de entrenamiento 
en el intervalo $$x=[-10, 10]$$, generado con los parámetros 
$$a=2.3$$ y $$b=-3$$. Importante no olvidar que los parámetros 
$$a=2.3$$ y $$b=-3$$ son desconocidos y los queremos estimar usando 
**Descenso por Gradiente** y también es importante mencionar que para
este problema en partícular es posible tener una solución analítica para 
estimar los parámetros. 

El primer paso es definir la función de error $$E(a, b \mid \mathcal X)$$,
en problemas de regresión una función de error viable es: $$E(a, b \mid \mathcal X) = \sum_{(x, y) \in \mathcal X} (y - f(x))^2$$.

La regla para actualizar los valores iniciales es: $$w = w - \eta \nabla_w E$$; 
por lo que se procede a calcular $$\nabla_w E$$ donde $$w$$ corresponde a los 
parámetros $$a$$ y $$b$$. 

$$
\begin{eqnarray*}
    \frac{\partial E}{\partial w} &=& \frac{\partial}{\partial w} \sum (y - f(x))^2 \\
    &=& 2 \sum (y - f(x)) \frac{\partial}{\partial w} (y - f(x)) \\
    &=& - 2 \sum (y - f(x)) \frac{\partial}{\partial w} f(x)
\end{eqnarray*}
$$

donde $$\frac{\partial}{\partial a} f(x) = x$$ y $$\frac{\partial}{\partial a} f(x) = 1$$.

Las ecuaciones para actualizar $$a$$ y $$b$$ serían:

$$
\begin{eqnarray*}
    e(y, x) &=& y - f(x) \\
    a &=& a + 2 \eta \sum_{(x, y) \in \mathcal X} e(y, x) x \\  
    b &=& b + 2 \eta \sum_{(x, y) \in \mathcal X} e(y, x)
\end{eqnarray*}
$$

Con el objetivo de visualizar descenso por gradiente y completar el ejemplo
anterior, el siguiente código implementa el proceso de optimización mencionado. 

Lo primero es generar el conjunto de entrenamiento y cargar las librerías 
necesarias. 

```python
import numpy as np
from matplotlib import pylab as plt
x = np.linspace(-10, 10, 50)
y = 2.3 * x - 3
```

El proceso inicia con valores aleatorios de $$a$$ y $$b$$, estos valores
podrían ser $$5.3$$ y $$-5.1$$, además se utilizará una $$\eta=0.0001$$,
se guardarán todos los puntos visitados en la lista $$D$$. Las variables y valores iniciales quedarían como:

```python
a = 5.3
b = -5.1
delta = np.inf
eta = 0.0001
D = [(a, b)]
``` 

El siguiente ciclo realiza la iteración del proceso de evolución y se 
detienen cuando los valores estimados varían poco entre dos iteraciones
consecutivas, en particular cuando en promedio el cambio en las 
constantes sea menor a $$0.0001$$. 


```python
while delta > 0.0001:
    hy = a * x + b
    e = (y - hy)
    a = a + 2 * eta * (e * x).sum()
    b = b + 2 * eta * e.sum()
    D.append((a, b))
    delta = np.fabs(np.array(D[-1]) - np.array(D[-2])).mean()
```

En la siguiente gráfica se muestra el camino que siguieron los parámetros
hasta llegar a los parámetros que generaron el problema, mostrados
en color naranja. 

![Ejemplo Descenso por Gradiente](/AprendizajeComputacional/assets/images/descenso.png)

## Regresión Logística

Recordando que en regresión logística la función que se desea minimizar es:

$$E(w, w_0 \mid \mathcal X) = - \sum_{(x, y) \in \mathcal X} y \log f(x) + (1-y) \log (1 -  f(x)),$$

donde $$ f(x) = \textsf{sigmoid}(w \cdot x + w_0) = \frac{1}{1 + \exp{-(w \cdot x + w_0)}}.$$

Tomando en cuenta que $$\frac{\partial \textsf{sigmoid}(g(x))}{\partial w} = \textsf{sigmoid}(g(x)) (1 - \textsf{sigmoid}(g(x))) \frac{\partial g(x)}{\partial w}.$$ Entonces la actualización de los coeficientes, 
donde $$v$$ corresponde a los parámetros $$w$$ y $$w_0$$, quedaría como:

$$
\begin{eqnarray*}
    \frac{\partial E}{\partial v} &=& - \sum \frac{y}{f(x)} \frac{\partial f(x)}{\partial v} + \frac{1 - y}{1 - f(x)} \frac{\partial (1 - f(x))}{\partial v} \\
    &=& - \sum \frac{y}{f(x)} \frac{\partial f(x)}{\partial v} + \frac{1 - y}{1 - f(x)} (- \frac{\partial f(x)}{\partial v}) \\
    &=& - \sum [\frac{y}{f(x)}  - \frac{1 - y}{1 - f(x)}] \frac{\partial f(x)}{\partial v} \\
    &=& - \sum \frac{y - yf(x) -f(x) + yf(x)}{f(x) (1 - f(x))} \frac{\partial f(x)}{\partial v}\\
    &=& -\sum \frac{y -f(x)}{f(x) (1 - f(x))} \frac{\partial f(x)}{\partial v}\\
    &=& -\sum \frac{y -f(x)}{f(x) (1 - f(x))} f(x) (1 - f(x))  \frac{\partial g(x)}{\partial v}\\
    &=& -\sum (y -f(x)) \frac{\partial g(x)}{\partial v}\\    
\end{eqnarray*}
$$

donde $$g(x) = w \cdot x + w_0$$, $$\frac{\partial E}{\partial w} = -\sum (y -f(x)) x $$ y  $$\frac{\partial E}{\partial w_0} = -\sum (y -f(x))$$.

# Propagación hacia Atrás

Propagación hacia atrás corresponde a la aplicación de la regla de la cadena,
es decir, $$\frac{\partial}{\partial w} f(g(x)) = f'(g(x)) \frac{\partial}{\partial w} g(x)$$ 

En el siguiente video se describe el algoritmo de propagación hacia atrás.

{%include propagacion.html %}

Realizando un ejemplo más complejo, en el siguiente video se 
muestra el procedimiento para estimar los parámetros de un modelo
de regresión logística. 

{%include propagacion2.html %}