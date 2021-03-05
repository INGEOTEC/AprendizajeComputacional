---
layout: default
title: Tipos de Aprendizaje
nav_order: 2
---

# Tipos de Aprendizaje Computacional
{: .fs-10 }


El **objetivo** de la unidad es conocer los diferentes tipos de aprendizaje computacional y los conceptos globales propios a este campo de estudio.

---

# Introducción

Aprendizaje computacional es una rama de la IA que estudia los algoritmos que son capaces de aprender a partir una serie de ejemplos o con alguna guía. Existen diferentes tipos de aprendizaje computacional los mas comunes son: aprendizaje supervisado, aprendizaje no-supervisado y aprendizaje por refuerzo. 

En **aprendizaje supervisado** se crean modelos partiendo de un conjunto de pares, entrada y salida, donde el objetivo es encontrar un modelo que logra aprender esta relación y logra predecir ejemplos no vistos en el proceso, en particular esto se le conoce como inductive learning. Complementando este tipo de aprendizaje supervisado se tiene lo que se conoce como transductive learning, en el cual se tiene un conjunto de pares y solamente se requiere conocer la salida en otro conjunto de datos. En este segundo tipo de aprendizaje todos los datos son conocidos en el proceso de aprendizaje. 

**Aprendizaje no-supervisado** es aquel donde se tiene un conjunto de entradas y se busca aprender alguna relación de estas entradas, por ejemplo, se podrían generar grupos o utilizar estas entradas para hacer un transformación o encontrar un patrón. 

Finalmente **aprendizaje por refuerzo** es aquel donde se aprender utilizando una función de refuerzo, es decir, se puede ver como un algoritmo que realiza una acción y la función de refuerzo regresa una calificación, de esta manera se puede saber cuáles son aquellas acciones que son mas recompensadas. En particular este último tipo de aprendizaje se puede entender mas claro con un ejemplo de control, ver el siguiente video. 

{%include refuerzo.html %}

## Aplicaciones de Aprendizaje Computacional

Es importante mencionar que no todo lo que se hace en aprendizaje automático está relacionado con juegos clásicos y que existen avances importantes en otros dominios. Como por ejemplo, recientemente en el área de cardiología se presenta el siguiente artículo [https://www.nature.com/articles/s41591-018-0268-3](), o en dermatología para detección de cancer [https://www.nature.com/articles/nature21056]() solo por mostrar algunos otros ejemplos. 

Cabe mencionar que los tres tipos de aprendizaje no son excluyentes uno del otro, comúnmente para resolver un problema complejo se combinan diferentes tipos de aprendizaje y otras tecnologías de IA para encontrar una solución aceptable. Probablemente una de las pruebas más significativas de lo que puede realizarse con aprendizaje automático es lo realizado por AlphaGo, leer el resumen del siguiente artículo [https://www.nature.com/articles/nature16961]() y recientemente se  quita una de las restricciones originales, la cual consiste en contar con un conjunto de jugadas realizadas por expertos, publicando este descubrimiento en [https://www.nature.com/articles/nature24270]().

Recientemente, en el área de aprendizaje, hay una tendencia de utilizar plataformas donde diferentes empresas u organismos gubernamentales o sin fines de lucro, ponen un problema e incentivas al publico en general a resolver este problema. La plataforma sirve de mediador en este proceso. Ver por ejemplo [https://www.kaggle.com]().

En el ámbito científico también se han generado este tipo de plataformas aunque su objetivo es ligeramente diferente, lo que se busca es tener una medida objetiva de diferentes soluciones y en algunos casos facilitar la reproducibilidad de las soluciones. Ver por ejemplo [http://codalab.org]().

# Aprendizaje No Supervisado

Iniciamos la descripción de los diferentes tipos de aprendizaje computacional con **aprendizaje no-supervisado**;
el cual inicia con un conjunto de elementos. Estos tradicionalmente se puede transformar a un conjunto de vectores en, i.e.
$$\mathcal X = \{ x_1, \ldots, x_N \}$$, donde $$x_i \in \mathbb R^d$$.
Durante este curso asumiremos que esta transformación existe y en algunos casos se hará explícito el algoritmo de transformación.

El **objetivo** en este tipo de aprendizaje es desarrollar algoritmos capaces de encontrar patrones en los datos, es decir, 
en  $$\mathcal X$$. Existen diferentes tareas que se pueden considerar dentro de aprendizaje no supervisado. Por ejemplo
el agrupamiento que puede servir para segmentar clientes o productos, en otra linea también cabría el análisis del carrito de 
compras (Market Basket Analysis); donde el objetivo es encontrar la co-ocurrencias de productos, es decir, se quiere estimar 
la probabilidad de que habiendo comprado un determinado artículo también se compre otro artículo. Con esta descripción ya se 
podrá estar imaginando la cantidad de aplicaciones en las que este tipo de algoritmos es utilizado en la actualidad. 

Regresando a la representación vectorial con una explicación abstracta, se puede visualizar que el conjunto de datos  $$\mathcal X$$ 
son los puntos que se muestran en la siguiente figura, claramente esto solo es posible si  $$x_i \in \mathbb R^2$$ o si
se hace algún tipo de transformación  $$f: \mathbb R^d \rightarrow \mathbb R^2$$.

![Puntos](/AprendizajeComputacional/assets/images/points.png) 

En la figura anterior se pueden observar dos o tres grupos de puntos, entonces el objetivo sería crear el algoritmo que dado 
 $$\mathcal X$$ no regrese un identificador por cada elemento, que represente al grupo al que pertenece el elemento en cuestión. 
Esta tarea se le conoce como agrupamiento (Clustering). Vamos asumir que aplicamos un algoritmo de agrupamiento a los datos 
anteriores; entonces, dado que podemos visualizar los datos, es factible representar el resultado del algoritmo si a
cada punto le asignamos un color dependiendo de la clase a la que pertenece. La siguiente figura muestra el resultado de este procedimiento. 

![Grupos](/AprendizajeComputacional/assets/images/cluster.png)
    
Se puede observar en la figura anterior, el algoritmo de agrupamiento separa los puntos en tres grupos, representados 
por los colores azul, naranja y verde. Cabe mencionar que utilizando algún otro criterio de optimización se hubiera podido encontrar 
dos grupos, el primero de ellos sería el grupo de los puntos de color verde y el segundo sería el grupo formado por los puntos 
azules y naranjas. Es importante recalcar que no es necesario visualizar los datos para aplicar un algoritmo de agrupamiento. 
En particular el ejercicio de visualización de datos y del resultado de agrupamiento que se muestra en la figuras anteriores 
tiene el objetivo de generarles una intuición de lo que está haciendo un algoritmo de agrupamiento. 


# Aprendizaje Supervisado


Aprendizaje supervisado es un problema donde el componente inicial es un conjunto de pares, entrada y salida. Es decir se cuenta
con  $$\mathcal X = \{ (x_1, y_1), \ldots, (x_N, y_N )\}$$, donde  $$x_i$$ corresponde a la  $$i$$-ésima entrada y  $$y_i$$ es
la salida asociada a esa entrada. Tomando en cuenta estas condiciones iniciales podemos definir el *objetivo*
de aprendizaje supervisado como encontrar un algoritmo capaz de regresar la salida  $$y$$ dada una entrada una entrada  $$x$$.

Existe una gran variedad de problemas que se puede categorizar como tareas de aprendizaje supervisado, solamente hay que recordar 
que en todos los casos se inicio con un conjunto  $$\mathcal X$$ de pares entrada y salida. En ocasiones la construcción del
conjunto es directa, por ejemplo supongamos que uno quiere identificar si una persona será sujeta a un crédito, entonces el 
conjunto a crear esta compuesto por las características de personas que se les ha otorgado un crédito y el estado final de
crédito, es decir, si el crédito fue pagado o no fue pagado. En otro ejemplo, supongamos que uno quiere crear un algoritmo 
que sea capaz de identificar si un texto dado tiene una polaridad positiva o negativa, entonces el conjunto se crea recolectando
textos y a cada texto un conjunto de personas decide si el texto dado es positivo o negativo y la polaridad final es el consenso 
de varias opiniones; a este problema en general se le conoce como análisis de sentimientos.

La cantidad de problema que se pueden poner en términos de aprendizaje supervisado es amplia, un problema tangible en esta época 
y relacionado a la pandemia del COVID-19 sería el crear un algoritmo que pudiera predecir cuántos serán los casos positivos
el día de mañana dando como entradas las restricciones en las actividades; por ejemplo escuelas cerradas, restaurantes a 30% 
de capacidad entre otras. 

Los ejemplos anteriores corresponden a dos de las clases de problemas que se resuelven en aprendizaje supervisado estas son problemas 
de clasificación y regresión. Definamos de manera formal estos dos problemas. Cuando  $$y \in \{0, 1\}$$ se dice que es un problema 
de **clasificación binaria**, por otro lado cuando  $$y \in \{0, 1\}^K$$ se encuentra uno en clasificación **multi-clase** o **multi-etiqueta**
y finalmente si  $$y \in \mathbb R$$ entonces es un problema de **regresión**.

Haciendo la liga entre los ejemplos y las definiciones anteriores, podemos observar que el asignar un crédito o la polaridad a 
un texto es un problema de clasificación binaria, dado que se puede asociar 0 y 1 a la clase positivo y negativo; y en el otro
caso a pagar o no pagar el crédito. Si el problema tiene mas categorías, supongamos que se desea identificar positivo, negativo o 
neutro, entonces se estaría en el problema de clasificación multi-clase. Por otro lado el problema de predecir
el número de casos positivos se puede considerar como un problema de regresión, dado que el valor a predecir 
dificilmente se podría considerar como una categoría.

Al igual que en aprendizaje no supervisado, en algunos casos es posible visualizar los elementos de  $$\mathcal X$$, el 
detalle adicional es que cada objeto tiene asociado una clase, entonces se selecciona un color para representar cada clase. En la siguiente
figura se muestra el resultado donde los elementos de  $$\mathcal X$$ se encuentra en  $$\mathbb R^2$$ y el color representa 
cada una de la clases de este problema de clasificación binaria.

![Conjunto de entrenamiento](/AprendizajeComputacional/assets/images/clases.png)

Usando esta representación es sencillo imaginar que el problema de clasificación se trata en encontrar una función que 
separe los puntos naranjas de los puntos azules, como se pueden imagina una simple linea recta podría separar estos puntos. 
La siguiente figura muestra un ejemplo de lo que haría un clasificador representado por la línea verde donde la clase es dada por 
el signo de  $$ax + b $$, donde  $$a$$ y  $$b$$ son parámetros identificados a partir de  $$\mathcal X$$.

![Clasificación](/AprendizajeComputacional/assets/images/clases2.png)

Siguiendo en esta misma linea, también es posible observar los puntos en un problema de regresión, solamente que 
en este caso un eje corresponde a las entradas, i.e.  $$x$$, y el otro eje es la salida, i.e.  $$y$$. 
La siguiente figura muestra un ejemplo de regresión, donde se puede observar que la idea es una encontrar una 
función que pueda seguir de manera adecuada los puntos datos.

![Regresión](/AprendizajeComputacional/assets/images/regresion.png)

El problema de regresión es muy conocido y seguramente ya se imaginaron que la respuesta sería encontrar los parámetros de una
parábola. La siguiente figura muestra una visualización del regresor, mostrado en color naranja, y los datos de entrenamiento en color azul

![Regresión](/AprendizajeComputacional/assets/images/regresion2.png)

Al igual que en aprendizaje no supervisado, este ejercicio de visualización no es posible en todos los problemas de aprendizaje supervisado, 
pero si permite ganar intuición sobre la forma en que trabajan estos algoritmos.