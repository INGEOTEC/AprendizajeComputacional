---
layout: default
title: Teoría de Decisión Bayesiana
nav_order: 3
---

# Teoría de Decisión Bayesiana
{: .fs-10 .no_toc }

El **objectivo** de la unidad es analizar el uso de la teoría de la probabilidad para 
la toma de decisiones. En particular el uso del teorema de Bayes para resolver problemas 
de clasificación y su uso para tomar la decisión que reduzca el riesgo. 

## Tabla de Contenido
{: .no_toc .text-delta }

1. TOC
{:toc}

---

# Introducción

Al diseñar una solución a un problema particular lo mejor que uno puede esperar es tener una 
certeza absoluta sobre la respuesta dada. Por ejemplo, si uno diseña un algoritmo que ordene 
un conjunto de números uno esperan que ese algoritmo siempre regrese
el orden correcto independientemente de la entrada dada, es mas un algoritmo de ordenamiento 
que en ocasiones se equivoca se consideraría de manera estricta erróneo.

Sin embargo, existen problemas cuyas características, como incertidumbre en la captura de los 
datos, variables que no se pueden medir, entre otros factores hacen que lo mejor que se puede 
esperar es un algoritmo exacto y preciso. Todos los problemas que
trataremos en Aprendizaje Computacional caen dentro del segundo escenario.

El lenguaje que nos permite describir de manera adecuada este tipo de ambiente, que se 
caracteriza por variables aleatorios es el de la probabilidad.

En particular podemos observar que el problema de regresión y de clasificación se puede 
definir utilizando la definición de probabilidad condicional

$$P(A \mid B) = \frac{P(AB)}{P(B)}$$,

la cual se entiende como la probabilidad de observar el evento $$A$$ sabiendo que ya se 
presentó el evento $$B$$.

Es sencillo visualizar que en estas condiciones, el problema de clasificación se puede 
plantear como el problema de selección la clase mas probable. Es decir, la clase seleccionada 
corresponde a $$\textsf{arg max}_i P(C_i \mid x)$$. Por ejemplo, en caso
de un problema binario, con clases $$0$$ y $$1$$ la respuesta sería:

$$ \begin{cases} 1, P(C=1 \mid x) > 0.5\\ 0, \text{de lo contrario} \end{cases} $$.

Por otro lado en regresión se puede plantear como 
$$p(y \mid x) = \mathcal N(g(x \mid \theta), \sigma^2)$$, donde la función $$g$$ y sus 
parámetros $$\theta$$ son identificados mediante el conjunto $$\mathcal X$$.

# Teorema de Bayes

Se puede observar que calcular la probabilidad conjunta de los eventos $$A$$ y $$B$$, i.e., 
$$P(AB)$$, en otras palabras el calcular las probabilidad conjunta entre entrada y salida. 
Para evitar este paso, se puede expresar la relación de probabilidad
condicional utilizando el teorema de Bayes.

En la siguientes lineas se deriva el teorema. Recordando que la probabilidad conjunta es 
conmutativa, es decir:$$P(AB) = P(BA)$$.

Tomando esta característica como base y utilizando la definición de probabilidad condicional 
se tiene:

$$P(A\mid B)P(B) = P(B \mid A) P(A)$$,

despejando $$P(A \mid B)$$ y asumiendo que $$P(B) > 0 $$ se obtiene el teorema de Bayes:

$$P(A \mid B) = \frac{P(B \mid A) P(A)}{P(B)}$$.

Haciendo un cambio de variables para dejarlo en términos de la clase $$C$$ y de la entrada 
$$x$$ se obtiene:

$$P(C \mid x) = \frac{P(x \mid C) P(C)}{P(x)}$$,

donde $$P(C)$$ es la probabilidad a priori, $$P(x \mid C)$$ es la verosimilitud de la clase y 
$$P(x)$$ es la evidencia. La cual se puede calcular utilizando la ley de la probabilidad total:

$$P(x) = \sum_i P(x \mid C_i) P(C_i)$$.

Finalmente, es importante mencionar que $$P(\cdot \mid x)$$ cumple con todos los axiomas de 
probabilidad, para este caso, suponiendo $$K$$ clases entonces $$\sum_i^K P(C_i \mid x) = 1$$.

# Riesgo

Como es de esperarse, existen aplicaciones donde el dar un resultado equivocado tiene un 
mayor impacto dependiendo de la clase. Por ejemplo, en un sistema de autentificación, 
equivocarse dándole acceso a una persona que no tiene permisos, es mucho mas
grave que no dejar entrar a una persona con los privilegios adecuados.

Una manera de incorporar el costo de equivocarse en el proceso de selección de la clase es 
modelarlo como una función de riesgo, es decir, seleccionar la clase que tenga el menor 
riesgo. Para realizar este procedimiento es necesario definir $$\alpha_i$$
como la acción que se toma al seleccionar la clase $$C_i$$. Entonces el riesgo esperado por 
tomar la acción $$\alpha_i$$ está definido por:

$$R(\alpha_i \mid x) = \sum_k \lambda_{ik} P(C_k \mid x) $$,

donde $$\lambda_{ik}$$ es el costo de tomar la acción $$i$$ en la clase $$k$$.

Suponiendo una función de costo $$0/1$$, donde el escoger la clase correcta tiene un costo 
$$0$$ y el equivocarse en cualquier caso tiene un costo $$1$$ se define como:

$$ \lambda_{ik} = \begin{cases} 0 \text{ si } i = k\\ 1 \text{ de lo contrario} \end{cases} $$.

Usando la función de costo $$0/1$$ el riesgo se define de la siguiente manera: 
$$R(\alpha_i \mid x) = \sum_k \lambda_{ik} P(C_k \mid x) = \sum_{k\neq i} P(C_k \mid x) = 1 - P(C_i \mid x) $$. Recordando que $$\sum_k P(C_k \mid x) = 1$$.

Por lo tanto en el caso de costo $$0/1$$ se puede observar que mínimo riesgo corresponde a la clase más probable.

## Acción nula

En algunas ocasiones es importante diseñar un procedimiento donde la acción a tomar sea el 
avisar que no se puede tomar una acción de manera automática y que se requiere una 
intervención manual.

La primera idea podría ser incrementar el número de clases y asociar una clase a la 
intervención manual, sin embargo en este procedimiento estaríamos incrementando la 
complejidad del problema. Un procedimiento mas adecuado sería incrementar el número
de acciones, $$\alpha$$ de tal manera que la acción $$\alpha_{K+1}$$ corresponda a la intervención esperada, esto para cualquier problema de $$K$$ clases.

La extensión del costo $$0,1$$ para este caso estaría definida como:

$$ \lambda_{ik} = \begin{cases} 0 \text{ si } i = k\\ 1 \text{ si } i = K + 1 \\ \lambda \text{ de lo contrario} \end{cases} $$,

donde $$0 < \lambda < 1$$.

Usando la definición de riesgo, el riesgo de tomar la acción $$\alpha_{K+1}$$ es $$R(\alpha_{K+1} \mid x) = \sum_k^K\lambda_{(K+1)k} P(C_k \mid x) = \sum_k^K \lambda P(C_k \mid x) = \lambda \sum_k^K P(C_k \mid x) = \lambda $$.

# Seleccionando la acción

Tomando en cuenta lo que hemos visto hasta el momento y usando como base el costo $$0,1$$ que 
incluye la acción nula, se puede observar que el riesgo de seleccionar una clase está dado 
por $$R(\alpha_i \mid x) = 1 - P(C_i \mid x) $$ y el riesgo de la acción nula es $$R(\alpha_{K+1} \mid x) = \lambda $$.

En esta circunstancias se selecciona la clase $$C_i$$ si es las clase con la probabilidad 
máxima, i.e., $$C_i = \textsf{arg max}_k P(C_k \mid x)$$ y además $$P(C_i \mid x) > 1 - \lambda $$.