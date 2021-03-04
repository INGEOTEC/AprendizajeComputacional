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
$\mathcal X = \{ x_1, \ldots, x_N \}$, donde $x_i \in \mathbb R^d $.
Durante este curso asumiremos que esta transformación existe y en algunos casos se hará explícito el algoritmo de transformación.

El **objetivo** en este tipo de aprendizaje es desarrollar algoritmos capaces de encontrar patrones en los datos, es decir, 
en $\mathcal X $. Existen diferentes tareas que se pueden considerar dentro de aprendizaje no supervisado. Por ejemplo
el agrupamiento que puede servir para segmentar clientes o productos, en otra linea también cabría el análisis del carrito de 
compras (Market Basket Analysis); donde el objetivo es encontrar la co-ocurrencias de productos, es decir, se quiere estimar 
la probabilidad de que habiendo comprado un determinado artículo también se compre otro artículo. Con esta descripción ya se 
podrá estar imaginando la cantidad de aplicaciones en las que este tipo de algoritmos es utilizado en la actualidad. 

Regresando a la representación vectorial con una explicación abstracta, se puede visualizar que el conjunto de datos $\mathcal X$ 
son los puntos que se muestran en la siguiente figura, claramente esto solo es posible si $x_i \in \mathbb R^2$ o si
se hace algún tipo de transformación $f: \mathbb R^d \rightarrow \mathbb R^2$.

![Puntos](/AprendizajeComputacional/assets/images/points.png) 

En la figura anterior se pueden observar dos o tres grupos de puntos, entonces el objetivo sería crear el algoritmo que dado 
$\mathcal X$ no regrese un identificador por cada elemento, que represente al grupo al que pertenece el elemento en cuestión. 
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