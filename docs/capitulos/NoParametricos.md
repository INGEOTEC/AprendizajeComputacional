---
layout: default
title: Métodos No Paramétricos
nav_order: 8
---

# Métodos No Paramétricos
{: .fs-10 }

El **objetivo** de la unidad es conocer las características de diferentes métodos no paramétricos y aplicarlos para 
resolver problemas de regresión y clasificación.

---

# Introducción

Los métodos paramétricos asumen que los datos provienen de un modelo común, esto da la ventaja de que el problema 
de estimar el modelo se limita a encontrar los parámetros del mismo, por ejemplo los parametros de una distribución 
gausiana. Por otro lado en los métodos no paramétricos asumen que datos similares se comportan de manera similar, 
estos algoritmos también se les conoces como algoitmos de memoría o basados en instancias.

# Clasificador de vecinos cercanos

El clasificador de vecinos cercanos es un clasificador simple de entender, la idea es utilizar el conjunto de entrenamiento y una función de distancia para asignar la clase de acuerdo a los k-vecinos más cercanos al objeto deseado.

{%include vecinos_cercanos.html %}

## KDtree

La manera más sencilla de crear el clasificador de vecinos cercanos es utilizando un método exhaustivo en el cálculo de distancia. Otra forma de realizar esto es mediante el uso de alguna estructura de datos que te permita el no realizar todas las operaciones. Uno de estos métodos puede ser KDTree. 

{%include kdtree.html %}

# Regresión

{%include vecinos_regresion.html %}