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