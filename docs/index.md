---
layout: default
title: Aprendizaje Computacional
nav_order: 1
permalink: /
---

# Aprendizaje Computacional
{: .fs-10 }

Mario Graff (mgraffg en ieee.org)
{: .fs-7 }

El **objetivo** es conocer las características de problemas de aprendizaje supervisado y no supervisado, y ser capaz de identificar las fortalezas y debilidades de diferentes tipos de algoritmos de aprendizaje de tal manera de que se pueda seleccionar el algoritmo mas adecuado al problema planteado.

---

# Introducción

El area de Inteligencia Artificial (IA) investiga la manera en que se puede crear/emular la inteligencia de manera artificial. Actualmente muchas de las formas de inteligencia artificial se prueban en una computadora, se puede llegar a pensar que la finalidad de IA es crear un programa inteligente, pero el área no se limita solamente a programas de computadora. Por otro lado, esta asignatura se encuentra dentro de la IA y tiene como objetivo el crear programas capaces de aprender. 

Una manera de entender qué es aprendizaje computacional es describiendo lo opuesto, de manera general un algoritmo (programa) es una secuencia de pasos que se siguen para resolver un problema, aunque esta secuencia no es evidente en algunos casos, podemos asumir que está ahí, por ejemplo, una hoja de cálculo o un procesador de texto son programas donde se sabe exactamente que se requiere y se puede generar un algoritmo para resolverlo. También cabría en esta descripción el algoritmo para encontrar la ruta mas corta entre dos nodos en un grafo, dicho algoritmo se encuentra en cualquier navegador gps. 

Por otro lado, existen problemas para los cuales escribir una secuencia de pasos para su solución es prácticamente imposible. Por ejemplo, identificar el correo no deseado, traducciones automáticas, o simplemente identificar el código postal en una carta ([ver](http://yann.lecun.com/exdb/mnist/)). Es en estos problemas donde recae el aprendizaje automático. 

# Notación

|Símbolo           | Significado                                              |
|------------------|----------------------------------------------------------|
|$$x$$             | Variable usada comunmente como entrada                   |
|$$y$$             | Variable usada comunmente como salida                    |
|$$\mathbb R$$     | Números reales                                           |
|$$\mathbf x$$     | Vector Columna $$\mathbf x \in \mathbb R^d$$             |
|$$d$$             | Dimensión                                                |
|$$\mathbf w^\intercal \cdot \mathbf x$$ | Producto punto donde $$\mathbf w$$ y $$\mathbf x \in \mathbb R^d$$ |
|$$\mathcal D$$    | Conjunto de datos                                        |
|$$\mathcal T$$    | Conjunto de entrenamiento                                | 
|$$\mathcal V$$    | Conjunto de validación                                   |
|$$\mathcal G$$    | Conjunto de prueba                                       |
|$$N$$             | Número de ejemplos                                       | 
|$$K$$             | Número de clases                                         |
|$$\mathbb P(\cdot)$$  | Probabilidad                                         |
|$$\mathcal X, \mathcal Y$$    | Variables aleatorías                         |
|$$\mathcal N(\mu, \sigma^2)$$    | Distribución Normal con parámetros $$\mu$$ y $$\sigma^2$$|
|$$f_{\mathcal X}$$| Función de densidad de probabilidad de $$\mathcal X$$    |
|$$\mathbb 1(e)$$     | Función para indicar; $$1$$ only if $$e$$ is true     |
|$$\Omega$$        | Espacio de búsqueda                                      |
|$$\mathbb V$$     | Varianza                                                 |
|$$\mathbb E$$     | Esperanza                                                |

# Desarrollo del Curso

Este curso ha evolucionado de las clases de Aprendizaje Computacional impartidas en la Maestría en Ciencia de Datos e Información ([MCDI](https://infotec.mx/MCDI)) de [INFOTEC](https://infotec.mx) y de Aprendizaje Computacional en la Maestría en Métodos para el Análisis de Políticas Públicas del [CIDE](http://cide.edu). 

En la MCDI compartí el curso con la Dra. [Claudia N. Sánchez](https://scholar.google.com.mx/citations?user=homoYl8AAAAJ&hl=es) y parte de este material,
en particular algunas figuras, fueron generadas por la Dra. Sánchez. 

# Bibliografía

El curso trata de ser auto-contenido, es decir, no debería de ser necesario leer otras fuentes para poder entenderlo y realizar las actividades. De cualquier manera es importante comentar que el curso está basado en los siguientes libros de texto:

- Introduction to machine learning, Third Edition. Ethem Alpaydin. MIT Press.
- [Probabilistic Machine Learning: An Introduction. Kevin Patrick Murphy. MIT Press](https://probml.github.io/pml-book/book1.html).
- An Introduction to Statistical Learning with Applications in R. Gareth James, Daniela Witten, Trevor Hastie, and Robert Tibshirani. Springer Texts in Statistics.
- All of Statistics. A Concise Course in Statistical Inference. Larry Wasserman. MIT Press.
- An Introduction to the Bootstrap. Bradley Efron and Robert J. Tibshirani. Monographs on Statistics and Applied Probability 57. Springer-Science+Business Media. 
- Understanding Machine Learning: From Theory to Algorithms. Shai Shalev-Shwartz and Shai Ben-David. Cambridge University Press.
