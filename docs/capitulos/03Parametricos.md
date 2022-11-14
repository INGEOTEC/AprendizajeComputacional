---
layout: default
title: Métodos Paramétricos
nav_order: 4
---

# Métodos Paramétricos
{: .fs-10 .no_toc }

El **objetivo** de la unidad es conocer las características de los modelos paramétricos y aplicar 
máxima verosimilitud para estimar los parámetros del modelo paramétrico en problemas de regresión y 
clasificación.

## Tabla de Contenido
{: .no_toc .text-delta }

1. TOC
{:toc}

---

# Introducción

Existen diferentes tipos de algoritmos que se puede utilizar para resolver problemas de
aprendizaje supervisado y no supervisado. En particular, esta unidad se enfoca en presentar 
las técnicas que se pueden caracterizar como métodos paramétricos. 

Los métodos paramétricos se identifican por asumir que los datos provienen de una distribución 
de la cual se desconocen los parámetros y el procedimiento es encontrar aquellos parámetros 
de la distribución que mejor modelen los datos. Una vez obtenidos los parámetros 
se cuenta con todos los elementos para utilizar el modelo y predecir la característica para
la cual fue entrenada. 

<!--
Habiendo descrito el uso de la probabilidad para identificar la clase mas probable o encontrar la 
acción con menor riesgo es posible describir como se entrenan y usan algunos algoritmos de 
clasificación y regresión basados en la teoría de probabilidad.

Antes de iniciar la descripción del algoritmos y las herramientas necesarias para su entrenamiento, es 
necesario introducir el uso de la función discriminante, $$g_i$$, para realizar la clasificación. 
Recordemos que la clase selección mediante la siguiente regla: $$\textsf{arg max}_i P(C_i \mid x) $$ o 
si se toma en cuenta el riesgo entonces se toma la acción que corresponde a $$\textsf{arg min}_i R
(\alpha_i \mid x)$$. En este sentido se puede incluir una función discriminante que sirva de puente 
entre el riesgo y/o la probabilidad, de tal manera que la clase seleccionada este dada por: $$\textsf{arg max}_i g_i(x)$$.

Se puede observar que la función discriminante, $$g_i$$, en el caso de probabilidad estaría definida 
como $$g_i(x) = P(C_i \mid x) $$ y en el caso de riesgo como: $$g_i(x) = - R(\alpha_i \mid x)$$.

Usando la función discriminante se puede inferir que no es necesario calcular la probabilidad 
posteriori para seleccionar la clase, esto porque la evidencia, es decir $$P(x)$$, es un factor común 
para todas las clases. Entonces una función discriminante equivalente estaría dada por: $$g_i(x) = P(x \mid C_i)P(C_i) $$.

Es momento para describir el procedimiento para encontrar los parámetros de $$P(x \mid C_i)$$ y $$P(C_i)$$.
-->

# Metodología

Hasta el momento se han presentado ejemplos de los pasos 4 y 5 
de la [metodología general](/AprendizajeComputacional/capitulos/01Tipos/#sec:metodologia-general);
esto fue en la sección de 
[Predicción](/AprendizajeComputacional/capitulos/02Teoria_Decision/#sec:prediccion-normal) 
y la sección de 
[Error de Clasificación](/AprendizajeComputacional/capitulos/02Teoria_Decision/#sec:error-clasificacion). 
Esta sección complementa los ejemplos anteriores al utilizar todos pasos 
de la metodología general de aprendizaje supervisado. En partícular se enfoca 
al paso 3 que corresponde al diseño del algoritmo $$f$$ que modela el fenómeno de
interés utilizando los datos $$\mathcal T \subset \mathcal D.$$

El algoritmo $$f$$ corresponde a asumir que los datos $$\mathcal D$$ provienen de 
una distribución $$F$$ la cual tiene una serie de parámetros $$\theta$$ que 
son identificados con $$\mathcal T.$$


# Estimación de Parámetros

Se inicia la descripción de métodos parámetricos presentando el procedimiento general
para estimar los parámetros de una distribución. Se cuenta con un conjunto $$\mathcal D$$
donde los elementos $$x \in \mathcal D$$ son $$x \in \mathbb R^d$$. Los elementos $$x \in \mathcal D$$ tienen un distribución $$F$$, i.e., $$x \sim F$$, son independientes y $$F$$ está 
definida por la función de densidad de probabilidad $$f_{\theta}$$ que está definida
por $$\theta$$ parámetros. Utilizando $$\mathcal D$$ el objetivo es identificar
los parámetros $$\theta$$ que harían observar a $$\mathcal D$$ lo más probable. 



<!--

Recordando que el ingrediente inicial en aprendizaje computacional es el conjunto de entrenamiento, 
formado por muestras independientes e identicamente distribuidas, se inicia la descripción desde el 
caso más simple enfocados a $$\mathcal X = \{ x_i \mid i=1, \ldots, N\} $$ y suponiendo que $$x_i$$ es 
tomada de una distribución de probabilidad $$p$$ definida por los parámetros $$\theta$$, es decir:

$$x_i \sim P(x \mid \theta) $$.

El objetivo es encontrar los parámetros $$\theta$$ que harían observar a $$\mathcal D$$ lo mas 
probable. 

Iniciando con una caso muy simple, donde $$\mathcal D=\{x_1, x_2\}$$. En este caso lo que se 
busca es maximizar la probabilidad de observar $$x_1$$ y $$x_2$$, es decir, $$\theta$$ es $$\textsf{arg max}_\theta \mathbb P(x_1, x_2 \mid \theta)$$. 

Utilizando la definición de probabilidad condicional, se 
puede escribir también como: $$P_\theta(x_1, x_1) = P_\theta(x_1 \mid x_2) P_\theta(x_2) $$ donde el 
parámetro $$\theta$$ se pone como subíndice para simplificar la notación.

Recordando que por definición $$x_1$$ y $$x_2$$ son independientes, entonces: $$P_\theta(x_1, x_1) = P_\theta(x_1 \mid x_2) P_\theta(x_2) = P_\theta(x_1) P_\theta(x_2) $$.

-->

## Verosimilitud

Una manera de plantear lo anterior es maximizando la verosimilitud. La versosimilitud es 
la distribución conjunta de los elementos en $$\mathcal D$$ tomandola como una función 
de los parámetros $$\theta,$$ es decir,

$$\mathcal L(\theta) = \prod_{x \in \mathcal D} f_\theta (x),$$

siendo el logaritmo de la verosimilitud 

$$\ell(\theta) = \log \mathcal L(\theta) = \sum_{x \in \mathcal D} \log f_\theta (x).$$

<!--
Extendiendo el caso anterior para todas las muestras en $$\mathcal X$$ queda la definición de 
verosimilitud que es:

$$l (\theta \mid \mathcal X) \equiv P(\mathcal X \mid \theta) = \prod_{i=1}^N P(x_i \mid \theta) $$.

Utilizando esta definición $$\theta$$ sería $$\textsf{arg max}_\theta l (\theta \mid \mathcal X)$$, 
por lo genera es mas sencillo trabajar con sumas en lugar de productos, por lo que una transformación 
muy utilizada es utilizar el logaritmo de la verosimilitud quedando la función a maximizar como:

$$\mathcal L(\theta \mid \mathcal X) = \sum_{i=1}^N \log P(x_i \mid \theta) $$.

-->

## Distribucción de Bernoulli

Habiendo definido la verosimilitud estamos en el momento de presentar el primer ejemplo para 
identificar $$\theta$$. Un problema de clasificación binaria se puede modelar con una distribución 
Bernoulli, suponiendo que una clase se representa como $$0$$ y la otra clase como $$1$$. Entonces, la 
probabilidad de ver $$1$$ es $$P(X=1) = p$$ y $$P(X=0) = 1 - p$$, donde $$p$$ es el parámetro a 
identificar. Combinando estas ecuaciones se obtiene que $$P(x) = p^x (1 - p)^{1-x}$$.

Utilizando el logaritmo de la verosimilitud se tiene:

$$\mathcal L(\mathcal p \mid \mathcal X) = \sum_{i=1}^N \log p^{x_i} (1 - p)^{1-x_i} = \sum_{i=1}^N x_i \log p + (1-x_i) \log (1 - p)$$.

Recordando que el máximo de $$\mathcal L(\mathcal p \mid \mathcal X) $$ se obtiene cuando $$\frac{d}{dp} \mathcal L(\mathcal p \mid \mathcal X) = 0$$. En estas condiciones estimar $$p$$ quedaría como:

$$ \begin{eqnarray} \frac{d}{dp} \mathcal L(\mathcal p \mid \mathcal X) &=& 0 \\ \frac{d}{dp} [ \sum_{i=1}^N x_i \log p + (1-x_i) \log (1 - p)] &=& 0 \\ \frac{d}{d p} [ \sum_{i=1}^N x_i \log p + \log (1 - p) (N - \sum_{i=1}^N x_i) ] &=& 0\\ \sum_{i=1}^N x_i \frac{d}{d p} \log \mathcal p + (N - \sum_{i=1}^N x_i) \frac{d}{d p} \log (1 - \mathcal p) &=& 0\\ \sum_{i=1}^N x_i \frac{1}{p} + (N - \sum_{i=1}^N x_i) \frac{-1}{(1 - p)} &=& 0\\ \end{eqnarray}$$

Realizando algunas operaciones algebraicas se obtiene:

$$\hat p = \frac{1}{N}\sum_{i=1}^N x_i $$.

## Distribución Multinomial

Para el caso de una clasificación multi-clase (de $$K$$ clases) una distribución adecuada sería 
Multinomial. Donde $$x_{i_k}$$ esta dada por:

$$x_{i_k} \begin{cases} 1 \text{ si el }i\text{-ésimo ejemplo es de clase } \\ 0 \text{ de lo contrario} \end{cases} $$.

Los parametros a estimar son: $$p_1, p_2, \ldots, p_K$$ con la restricción $$\sum_{k=1}^K p_k = 1$$. 
El parámetro estimador quedaría como:

$$p_k = \frac{1}{N} \sum_{i=1}^N x_{i_k} $$.

## Distribución Normal

Para el caso de $$P(x \mid C_i)$$ una de las distribuciones mas utilizadas es la normal, la cuál está 
identificada por dos parámetros: $$\mu$$ y $$\sigma$$ en el caso de $$x \in \mathbb R$$; y $$ \mathbb \mu $$ y $$\Sigma$$ en el caso de $$x \in \mathbb R^d $$. Recordando que $$\sum_x P(x \mid C_i) = 1 $$, es decir, $$P(\cdot \mid C_i) $$ obedece los axiomas de probabilidad. Entonces, para un problema de $$ K $$ clases se tienen $$ K $$ pares de parámetros a identificar. Una manera de visualizar esto 
sería por segmentar el conjunto de entrenamiento de tal manera que $$\mathcal X_c = \{x \mid (x, y) \in \mathcal X, y=c \} $$.

Entonces $$\mathcal L(\mu_k, \sigma_k \mid \mathcal X_k) $$ sería la verosimilitud para identificar 
los parámetros correspondientes a la clase $$k$$.

Los estimados de $$\mu$$ y $$\sigma$$ se obtendrían como:

$$m_k = \frac{1}{\mid \mathcal X_k \mid} \sum_{x \in \mathcal X_k} x $$

$$s^2_k = \frac{1}{\mid \mathcal X_k \mid} \sum_{x \in \mathcal X_k} (x - m_k)^2 $$,

para el caso de $$x \in \mathbb R$$ o equivalente para el caso de que las variables sea independientes 
en el caso de $$x \in \mathbb R^d$$, como se verá en los siguientes videos.

# Clasificador Bayesiano Ingenuo

En esta sección se describe mediante videos el desarrollo de un clasificador Bayesiano 
ingenuo, pero antes de iniciar con la descripción lo primero es explicar el ejemplo 
que se utilizará. 

En la siguiente figura se muestra un conjunto de puntos en $$ \mathbb{R}^2 $$ los cuales tiene asociado un color. El objetivo es encontrar una función capaz de definir el color de un nuevo punto, $$ x \in \mathbb{R}^2 $$, dado.

![Ejemmplo](/AprendizajeComputacional/assets/images/clusters.png) 

El primer paso para hacer el ejemplo autocontenido es mostrar como se generó la figura, lo cual se puede ver en el siguiente video.

{%include problema_clasificacion.html %}

El algoritmo de clasificación se describe en el siguiente video. 

{%include naive_bayes.html %}

# Regresión

Hasta este momento se han revisado métodos paramétricos en clasificación, ahora es el turno de abordar 
esto en el problema de regresión.

Recordando, regresión es un problema de aprendizaje supervisado, es decir se cuenta con un conjunto de 
entrenamiento, $$\mathcal X = \{ (x_1, y_1), \ldots, (x_N, y_N )\}$$, de pares entrada y salida; la 
salida es $$ y_i \in \mathbb R$$.

Entonces se busca una función con la forma $$ f: \mathbb{ R^d } \rightarrow \mathbb R $$ y que se 
comporte como: $$ \forall_{(x, y) \in \mathcal X} f(x) = y  + \epsilon $$. 

Este problema se puede plantear como un problema de optimización o como un problema de algebra lineal. 
Viéndolo como un problema de algebra lineal lo que se tiene es 

$$ X w = y $$

donde $$ X $$ son las observaciones, entradas o combinación de entradas, $$ w $$, son los pesos 
asociados y $$y$$ es el vector que contiene las variables dependientes. 

Tanto $$X$$ como $$y$$ son datos que se obtienen de $$\mathcal X$$ entonces lo que hace falta 
identificar es $$w$$, lo cual se puede realizar de la siguiente manera

$$ X^T X w = X^T y $$

donde $$X^T$$ es la transpuesta de $$X$$. Despejando $$w$$ se tiene

$$w = (X^T X)^{-1} X^T y.$$

En el siguiente video se muestra un ejemplo de regresión:

{%include regresion.html %}