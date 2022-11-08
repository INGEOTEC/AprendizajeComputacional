---
layout: default
title: Tipos de Aprendizaje
nav_order: 2
---

# Tipos de Aprendizaje Computacional
{: .fs-10 .no_toc }

El **objetivo** de la unidad es conocer los diferentes tipos de aprendizaje computacional y los conceptos globales propios a este campo de estudio.

## Tabla de Contenido
{: .no_toc .text-delta }

1. TOC
{:toc}

---

# Introducción

Aprendizaje computacional es una rama de la IA que estudia los algoritmos que son capaces de aprender a partir de una serie de ejemplos o con alguna guía. Existen diferentes tipos de aprendizaje computacional, los mas comunes son: aprendizaje supervisado, aprendizaje no-supervisado y aprendizaje por refuerzo. 

En **aprendizaje supervisado** se crean modelos partiendo de un conjunto de pares, entrada y salida, donde el objetivo es encontrar un modelo que logra aprender esta relación y predecir ejemplos no vistos en el proceso, en particular a esto se le conoce como _inductive learning_. Complementando este tipo de aprendizaje supervisado se tiene lo que se conoce como _transductive learning_, en el cual se cuenta con un conjunto de pares y solamente se requiere conocer la salida en otro conjunto de datos. En este segundo tipo de aprendizaje todos los datos son conocidos en el proceso de aprendizaje. 

**Aprendizaje no-supervisado** es aquel donde se tiene un conjunto de entradas y se busca aprender alguna relación de estas entradas, por ejemplo, generando grupos o utilizando estas entradas para hacer una transformación o encontrar un patrón. 

Finalmente **aprendizaje por refuerzo** es aquel donde se aprender utilizando una función de refuerzo, es decir, se puede ver como un algoritmo que realiza una acción y la función de refuerzo regresa una calificación, de esta manera se puede saber cuáles son aquellas acciones que tienen una mayor recompensa. En particular este último tipo de aprendizaje se puede entender mas claro con un ejemplo de control, como se muestra en el siguiente video. 

{%include refuerzo.html %}

## Aplicaciones de Aprendizaje Computacional

Es importante mencionar que no todo lo que se hace en aprendizaje automático está relacionado con juegos clásicos y que existen avances importantes en otros dominios. Como por ejemplo, recientemente en el área de cardiología se presenta el siguiente artículo [https://www.nature.com/articles/s41591-018-0268-3](), o en dermatología para detección de cancer [https://www.nature.com/articles/nature21056]() solo por mostrar algunos otros ejemplos. 

Cabe mencionar que los tres tipos de aprendizaje no son excluyentes uno del otro, comúnmente para resolver un problema complejo se combinan diferentes tipos de aprendizaje y otras tecnologías de IA para encontrar una solución aceptable. Probablemente una de las pruebas más significativas de lo que puede realizarse con aprendizaje automático es lo realizado por AlphaGo, leer el resumen del siguiente artículo [https://www.nature.com/articles/nature16961]() y recientemente se  quita una de las restricciones originales, la cual consiste en contar con un conjunto de jugadas realizadas por expertos, publicando este descubrimiento en [https://www.nature.com/articles/nature24270]().

En el área de aprendizaje, hay una tendencia de utilizar plataformas donde diferentes empresas u organismos gubernamentales o sin fines de lucro, ponen un problema e incentivan al publico en general a resolver este problema. La plataforma sirve de mediador en este proceso. Ver por ejemplo [https://www.kaggle.com]().

En el ámbito científico también se han generado este tipo de plataformas aunque su objetivo es ligeramente diferente, lo que se busca es tener una medida objetiva de diferentes soluciones y en algunos casos facilitar la reproducibilidad de las soluciones. Ver por ejemplo [http://codalab.org]().

# Aprendizaje No Supervisado

Iniciamos la descripción de los diferentes tipos de aprendizaje computacional con **aprendizaje no-supervisado**; el cual inicia con un conjunto de elementos. Estos tradicionalmente se puede transformar en conjunto de vectores en, i.e. $$\mathcal D = \{ x_1, \ldots, x_N \}$$, donde $$x_i \in \mathbb R^d$$. Durante este curso asumiremos que esta transformación existe y en algunos casos se hará explícito el algoritmo de transformación.

El **objetivo** en aprendizaje no supervisado es desarrollar algoritmos capaces de encontrar patrones en los datos, es decir, en $$\mathcal D$$. Existen diferentes tareas que se pueden considerar dentro de este tipo de aprendizaje. Por ejemplo, el agrupamiento puede servir para segmentar clientes o productos, en otra linea también cabría el análisis del carrito de compras (Market Basket Analysis); donde el objetivo es encontrar la co-ocurrencias de productos, es decir, se quiere estimar la probabilidad de que habiendo comprado un determinado artículo también se compre otro artículo. Con esta descripción ya se podrá estar imaginando la cantidad de aplicaciones en las que este tipo de algoritmos es utilizado en la actualidad. 

Regresando a la representación vectorial, existen casos donde se pueden visualizar los elementos de $$\mathcal D$$, lo cuales están representados como puntos que se muestran en la siguiente figura. Claramente esto solo es posible si $$x_i \in \mathbb R^2$$ o si se hace algún tipo de transformación  $$f: \mathbb R^d \rightarrow \mathbb R^2$$.

![Puntos](/AprendizajeComputacional/assets/images/points.png) 

En la figura anterior se pueden observar dos o tres grupos de puntos, entonces el objetivo sería crear el algoritmo que dado $$\mathcal D$$ regrese un identificador por cada elemento, dicho identificador representa el grupo al que pertenece el elemento en cuestión. Esta tarea se le conoce como agrupamiento (Clustering). Asumiendo que se aplica un algoritmo de agrupamiento a los datos anteriores; entonces, dado que podemos visualizar los datos, es factible representar el resultado del algoritmo si a
cada punto se le asigna un color dependiendo de la clase a la que pertenece. La siguiente figura muestra el resultado de este procedimiento. 

![Grupos](/AprendizajeComputacional/assets/images/cluster.png)
    
Se puede observar en la figura anterior, el algoritmo de agrupamiento separa los puntos en tres grupos, representados por los colores azul, naranja y verde. Cabe mencionar que utilizando algún otro criterio de optimización se hubiera podido encontrar dos grupos, el primero de ellos sería el grupo de los puntos de color verde y el segundo sería el grupo formado por los puntos azules y naranjas. Es importante recalcar que no es necesario visualizar los datos para aplicar un algoritmo de agrupamiento. En particular el ejercicio de visualización de datos y del resultado de agrupamiento que se muestra en la figuras anteriores tiene el objetivo de generar una intuición de lo que está haciendo un algoritmo de agrupamiento. 

# Aprendizaje Supervisado
{: #sec:aprendizaje-supervisado }

Aprendizaje supervisado es un problema donde el componente inicial es un conjunto de pares, entrada y salida. Es decir se cuenta con $$\mathcal D = \{ (x_1, y_1), \ldots, (x_N, y_N )\}$$, donde $$x_i \in \mathbb R^d$$ corresponde a la  $$i$$-ésima entrada y $$y_i$$ es la salida asociada a esa entrada. Tomando en cuenta estas condiciones iniciales podemos definir el *objetivo* de aprendizaje supervisado como encontrar un algoritmo capaz de regresar la salida $$y$$ dada una entrada una entrada $$x$$.

Existe una gran variedad de problemas que se puede categorizar como tareas de aprendizaje supervisado, solamente hay que recordar que en todos los casos se inicia con un conjunto $$\mathcal D$$ de pares entrada y salida. En ocasiones la construcción del conjunto es directa, por ejemplo en el caso de que se quiera identificar si una persona será sujeta a un crédito, entonces el 
conjunto a crear esta compuesto por las características de las personas que se les ha otorgado un crédito y el estado final de crédito, es decir, si el crédito fue pagado o no fue pagado. En otro ejemplo, suponiendo que se quiere crear un algoritmo capaz de identificar si un texto dado tiene una polaridad positiva o negativa, entonces el conjunto se crea recolectando textos y a cada texto un conjunto de personas decide si el texto dado es positivo o negativo y la polaridad final es el consenso de varias opiniones; a este problema en general se le conoce como análisis de sentimientos.

La cantidad de problema que se pueden poner en términos de aprendizaje supervisado es amplia, un problema tangible en esta época y relacionado a la pandemia del COVID-19 sería el crear un algoritmo que pudiera predecir cuántos serán los casos positivos el día de mañana dando como entradas las restricciones en las actividades; por ejemplo escuelas cerradas, restaurantes al 30% de capacidad entre otras. 

Los ejemplos anteriores corresponden a dos de las clases de problemas que se resuelven en aprendizaje supervisado estas son problemas de clasificación y regresión. Definamos de manera formal estos dos problemas. Cuando  $$y \in \{0, 1\}$$ se dice que es un problema de **clasificación binaria**, por otro lado cuando  $$y \in \{0, 1\}^K$$ se encuentra uno en clasificación **multi-clase** o **multi-etiqueta** y finalmente si  $$y \in \mathbb R$$ entonces es un problema de **regresión**.

Haciendo la liga entre los ejemplos y las definiciones anteriores, podemos observar que el asignar un crédito o la polaridad a un texto es un problema de clasificación binaria, dado que se puede asociar 0 y 1 a la clase positivo y negativo; y en el otro caso a pagar o no pagar el crédito. Si el problema tiene mas categorías, supongamos que se desea identificar positivo, negativo o neutro, entonces se estaría en el problema de clasificación multi-clase. Por otro lado el problema de predecir el número de casos positivos se puede considerar como un problema de regresión, dado que el valor a predecir 
difícilmente se podría considerar como una categoría.

Al igual que en aprendizaje no supervisado, en algunos casos es posible visualizar los elementos de  $$\mathcal D$$, el detalle adicional es que cada objeto tiene asociado una clase, entonces se selecciona un color para representar cada clase. En la siguiente figura se muestra el resultado donde los elementos de  $$\mathcal D$$ se encuentra en $$\mathbb R^2$$ y el color representa cada una de la clases de este problema de clasificación binaria.

![Conjunto de entrenamiento](/AprendizajeComputacional/assets/images/clases.png)

Usando esta representación es sencillo imaginar que el problema de clasificación se trata en encontrar una función que separe los puntos naranjas de los puntos azules, como se pueden imagina una simple linea recta podría separar estos puntos. La siguiente figura muestra un ejemplo de lo que haría un clasificador representado por la línea verde; la clase es dada por el signo de  $$ax + by + c $$, donde  $$a$$, $$b$$ y  $$c$$ son parámetros identificados a partir de  $$\mathcal D$$.

![Clasificación](/AprendizajeComputacional/assets/images/clases2.png)

Siguiendo en esta misma linea, también es posible observar los puntos en un problema de regresión, solamente que en este caso un eje corresponde a las entradas, i.e.  $$x$$, y el otro eje es la salida, i.e. $$y$$. La siguiente figura muestra un ejemplo de regresión, donde se puede observar que la idea es una encontrar una función que pueda seguir de manera adecuada los puntos datos.

![Regresión](/AprendizajeComputacional/assets/images/regresion.png)

El problema de regresión es muy conocido y seguramente ya se imaginaron que la respuesta sería encontrar los parámetros de una parábola. La siguiente figura muestra una visualización del regresor, mostrado en color naranja y los datos de entrenamiento en color azul

![Regresión](/AprendizajeComputacional/assets/images/regresion2.png)

Al igual que en aprendizaje no supervisado, este ejercicio de visualización no es posible en todos los problemas de aprendizaje supervisado, pero si permite ganar intuición sobre la forma en que trabajan estos algoritmos.

## Definiciones de Aprendizaje Supervisado

El primer paso es empezar a definir los diferentes conjuntos con los 
que se trabaja en aprendizeje computacional. Todo inicia 
con el **conjunto de entrenamiento** identificado en este 
documento como $$\mathcal T$$. Este conjunto se utiliza 
para estimar los parámetros o en general buscar 
un algoritmo que tenga el comportamiento esperado.

Se puede asumir que existe una función $$f$$ que genera la relación entrada salida mostrada en $$\mathcal D$$, es decir, idealmente se tiene que $$\forall_{(x, y) \in \mathcal D} f(x) = y$$. En este contexto, aprendizaje supervisado se entiende como el proceso de encontrar una función $$h^*$$ que se comporta similar a $$f$$.

Para encontrar $$h^*$$, se utiliza $$\mathcal T$$; el conjunto de hipótesis (funciones), $$\mathcal H$$, que se considera puede aproximar $$f$$; una función de error, $$L$$; y el error empírico $$E(h \mid \mathcal T) = \sum_{(x, y) \in \mathcal T} L(y, h(x))$$. Utilizando estos elementos la función buscada es: $$h^* = \textsf{argmin}_{h \in \mathcal{H}} E(h \mid \mathcal T)$$.

El encontrar la función $$h^*$$ no resuelve el problema de aprendizaje en su totalidad, además se busca una función que sea capaz de generalizar es decir de predecir correctamente instancias no vistas. Considerando que se tiene un conjunto de prueba, $$\mathcal G=\{(x_i, y_i)\}$$ para $$i=1 \ldots M$$, donde $$\mathcal D \cap \mathcal G = \emptyset$$. La idea es que el error empírico sea similar en el conjunto de entrenamiento y prueba. Es decir $$E(h^* \mid \mathcal D) \approx E(h^* \mid \mathcal G) $$.

### Características de la hipótesis

Continuando con algunas definiciones, en la búsqueda de encontrar $$h^*$$ uno puede elegir por aquella que captura todos los datos de entrenamiento siendo muy específica. Para poder explicar mejor este concepto usemos la siguiente figura que representa un ejemplo sintético.

![Hipótesis mas específica](/AprendizajeComputacional/assets/images/clases-sh.png)

Los puntos naranja representan una clase y los puntos azules son la otra clase, el clasificador es mostrado en el rectángulo. Para completar el funcionamiento de este clasificador falta mencionar que cualquier nuevo punto que esté dentro del rectángulo será calificado como clase azul y naranja si se encuentra fuera del rectángulo. Como se puede observar, el rectángulo contiene todos los elementos de una clase y en alguno de sus lados toca con uno de los puntos del conjunto de entrenamiento de la clase data.

En este momento, es posible visualizar que el complemento de ser muy específico es ser lo mas general posible. La siguiente siguiente figura muestra un clasificador que es lo mas general. Se observa que el rectángulo casi toca uno de los puntos del conjunto de entrenamiento de la clase contraria, esto lo hace del lado exterior y el procedimiento para clasificar continua siendo el que se mencionó anteriormente.

![Hipótesis más general](/AprendizajeComputacional/assets/images/clases-gh.png)

Uno puede elegir una hipótesis que se encuentre entre la hipótesis mas general y la más específica, esto también se puede visualizar en la siguiente figura. Donde todas las hipótesis se encuentran representadas en gris.

![Clase de hipótesis](/AprendizajeComputacional/assets/images/clases-ch.png)

Finalmente, para poder describir mejor el comportamiento de un clasificador se hace uso de la distancia que hay entre la hipótesis mas general y específica y la hipótesis utilizada. El margen se puede visualizar en la siguiente figura, donde la hipótesis más general es mostrada en verde, la más específica en morada y la hipótesis utilizada en negro.

![Marger](/AprendizajeComputacional/assets/images/clases-margen.png)

### Estilos de Aprendizaje

Utilizando $$\mathcal T$$ y $$\mathcal G$$ podemos definir **inductive learning** como el proceso de aprendizaje en donde solamente se utiliza $$\mathcal T$$ y el algoritmo debe de ser capaz de predecir cualquier instancia. Por otro lado, **transductive learning** es el proceso de aprendizaje donde se utilizar $$\mathcal T \cup \{ x \mid (x, y) \in \mathcal G \}$$ para aprender y solamente es de interés el conocer la clase o variable dependiente del 
conjunto $$\mathcal G$$.

### Sobre-aprendizaje

Existen clases de algoritmos, $$\mathcal H$$, que tienen un mayor grado de libertad el cual se ve reflejado en una capacidad superior para aprender, pero por otro lado, existen problemas donde no se requiere tanta libertad, esta combinación se traduce en que el algoritmo no es capaz de generalizar y cuantitativamente se ve como $$E(h^* \mid \mathcal T) \ll E(h^* \mid \mathcal G)$$.

Para mostrar este caso hay que imaginar que se tiene un algoritmo que guarda el conjunto de entrenamiento y responde lo siguiente:

$$ h^*(x) = \begin{cases} y & \text{si} (x, y) \in \mathcal T\\ 0 & \end{cases} $$

Es fácil observar que este algoritmo tiene $$E(h^* \mid \mathcal T) = 0$$ dado que se aprende todo el conjunto de entrenamiento.

La siguiente figura muestra el comportamiento de un algoritmo que sobre-aprende, el algoritmo se muestra en la linea naranja, la linea azul corresponde a una parábola (cuyos parámetros son identificados con los datos de entrenamiento) y los datos de entrenamiento no se muestran; pero se pueden visualizar dado que son datos generados por una parábola mas un error gaussiano. Entonces podemos ver que la linea naranja pasa de manera exacta por todos los datos de entrenamiento y da como resultado la linea naranja que claramente tiene un comportamiento mas complejo que el comportamiento de la parábola que generó los datos.

![Sobre-entrenamiento](/AprendizajeComputacional/assets/images/overfitting-2.png)

### Sub-aprendizaje

Por otro lado existen problemas donde el conjunto de algoritmos $$\mathcal H $$ no tienen los grados de libertad necesarios para aprender, dependiendo de la medida de error esto se refleja como $$E(h^* \mid \mathcal T) \gg 0$$.

La siguiente figura muestra un problema de clasificación donde el algoritmo de aprendizaje presenta el problema de sub-aprendizaje. El problema de clasificación es encontrar una función que logre separar las dos clases, las cuales están representadas por los puntos en color azul y verde, el algoritmo de aprendizaje intenta separar estas clases mediante una linea (dibujada en rojo) y como se puede observar una linea no tiene los suficientes grados de libertad para separar las clases.

![Sub-entrenamiento](/AprendizajeComputacional/assets/images/sub-aprendizaje.png)