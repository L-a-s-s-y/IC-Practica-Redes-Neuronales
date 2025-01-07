# Memoria de la práctica de redes neuronales
#### José Antonio Laserna Beltrán
## Introducción
A continuación se expone la práctica de redes neuronales de la asignatura de inteligencia computacional. Se hará un recorrido sobre el lenguaje de programación utilizado junto a sus bibliotecas y sobre como ha sido el proceso de refinamiento de la solución.

Los experimentos realizados en un primer momento se realizaron utilizando la herramienta [Colab](https://colab.research.google.com/) de Google. Sin embargo, posteriormente se comenzaron a realizar en local debido a la lentitud de la herramienta y la disponibilidad de una GPU para acelerar el entrenamiento. Las características de la máquina utilizada para el desarrollo de la practica son:
- CPU: Intel® Core™ i7-8750H
- GPU: NVIDIA GeForce GTX 1050 Ti
- Fedora Linux 40 (Workstation Edition)

## Lenguaje de programación
Para la realización de la práctica se ha empleado el lenguaje de programación **Python**. Esta elección se debe a la existencia de los módulos `tensorflow`, `skicit-learn`, `numpy` y `matplotlib` que facilitan la resolución de la práctica. Como puede deducirse de esta introducción, no se ha realizado una implementación propia de los algoritmos de aprendizaje de las redes neuronales.
## Módulos empleados
- [tensorflow](https://www.tensorflow.org/)
- [keras](https://keras.io/)
- [skicit-learn](https://scikit-learn.org/stable/index.html)
- [numpy](https://numpy.org/)
- [matplotlib](https://matplotlib.org/)

## Definiciones
#### Capa completamente conectada
Una capa de neuronas completamente conectadas es una capa de neuronas en las que cada una de ellas tiene conexiones (con sus respectivos pesos asociados) con todas las neuronas de la capa anterior y con las capa siguiente.
#### Capa convolucional
Una capa convolucional es una capa de neuronas en las que se aplica una convolución a los datos entrantes. Estos datos deben estar en forma matricial para aplicar esta convolución. Una convolución es una operación que se le aplica a un elemento de una matriz en la que influyen los elementos vecinos del elemento objeto de la operación.
#### Padding
Consiste en, a la hora de realizar una convolución, añadir elementos a los valores vecinos ficticios a los elementos extremos de la matriz si estos no cuentan con suficientes vecinos, de forma que estos también puedan verse afectados de por la convolución.
#### Kernel
El kernel define el tamaño de la convolución. En el caso de la práctica se está utilizando un kernel de tamaño 3, lo que significa que a la hora de hacer la convolución de un elemento se tendrá en cuenta una matriz de 3x3 vecinos para el elemento.
#### Pooling
Es una operación que reduce la dimensionalidad de los datos reduciendo el número de entradas para la siguiente capa de la red neuronal. En nuestro caso de utiliza un pooling de 2x2, lo que implica que se dividiría la matriz de datos en submatrices de tamaño 2x2 y, de esas submatrices se toma un único elemento.
#### Batch Normalization
Se utiliza para normalizar la salida (y la entrada) de los datos entre capas de la red neuronal.
#### Early Stopping
Es una técnica para evitar el sobreaprenzaje de la red. Cuando se detecta que no hay una mejora en la partición de validación se detiene el aprendizaje del entrenamiento.
#### Dropout
Otra técnica para evitar el sobre aprendizaje de la red. Consiste en desconectar de manera aleatoria neuronas de la red, evitando su ajuste durante una época.
#### Partición de validación
Es una sección del conjunto de entrenamiento que se secciona de él y se utiliza para como forma de monitorear el entrenamiento de forma separada.
#### Weight decay
Técnica que penaliza los pesos grandes. Hay distintas formas de aplicación de esta técnica.
## Primera red neuronal
La primera red neuronal que se implementó fue sencilla. Constaba de una única capa oculta de 256 neuronas completamente conectada con función de activación sigmoide y una capa de salida de 10 neuronas con función de activación softmax. Se utilizaba como optimizador *Adam*, como función de pérdida *categorical crossentropy* y 100 épocas.

El tiempo de entrenamiento era de unos 4 minutos y los resultados eran los siguientes:
- Sobre el conjunto de entrenamiento: `Precisión: 0.96705 Tasa de error: 0.032950000000000035`
- Sobre el conjunto de prueba: `Precisión: 0.9602 Tasa de error: 0.03979999999999995`

Se realizaron experimentos con distintos tamaños de neuronas, especialmente con mayores números de ellas. En concreto con 512 neuronas estos eran los resultados:
- Sobre el conjunto de entrenamiento: `Precisión: 0.9706833333333333 Tasa de error: 0.029316666666666658`
- Sobre el conjunto de prueba: `Precisión: 0.9651 Tasa de error: 0.03490000000000004`

~~~
#RED 1: Completamente conectada | 1 CAPA
model_red_1 = tf.keras.models.Sequential()
model_red_1.add(tf.keras.layers.Dense(256, activation='sigmoid'))
model_red_1.add(tf.keras.layers.Dense(10, activation='softmax'))
#Para entrenar el modelo
#tf.keras.optimizers.Adam(learning_rate=0.0015)
model_red_1.compile('adam', 'categorical_crossentropy') #funcion de perdida
etiquetas_red_1 = tf.keras.utils.to_categorical(etiquetas_train)
model_red_1.fit(images_train, etiquetas_red_1, epochs=100)

preds_train_red_1 = model_red_1.predict(images_train).argmax(axis=1)
acc_train = accuracy_score(etiquetas_red_1.argmax(axis=1),preds_train_red_1)
print("Precisión: ", acc_train, "Tasa de error: ", 1 - acc_train)

preds_test_red_1 = model_red_1.predict(images_test).argmax(axis=1)
acc_test = accuracy_score(etiquetas_test, preds_test_red_1)
print("Precisión: ", acc_test, "Tasa de error: ", 1 - acc_test)
~~~

## Segunda red neuronal
Esta red fue realmente una consecuencia lógica de la primera. Consistió en la adición de capas ocultas. En concreto se utilizaron 3 capas completamente conectadas de 256, 128 y 64 neuronas respectivamente, más la capa de salida de 10 neuronas. Las funciones de activación utilizadas siguieron siendo análogas a las de la primera red neuronal. Se utilizaba como optimizador *Adam*, como función de pérdida *categorical crossentropy* y 100 épocas.

Los resultados con esta versión de la red eran los siguientes con un tiempo de entrenamiento de uno 4 minutos:
- Sobre el conjunto de entrenamiento: `Precisión: 0.9824166666666667 Tasa de error: 0.017583333333333284`
- Sobre el conjunto de prueba: `Precisión: 0.9705 Tasa de error: 0.02949999999999997`

Se probó a aumentar el tamaño de la red con una capa adicional al de 512 situada antes de la capa de 256 neuronas. Los resultados con esta versión de la red eran los siguientes con un tiempo de entrenamiento de uno 4 minutos:
- Sobre el conjunto de entrenamiento: `Precisión: 0.9868333333333333 Tasa de error: 0.01316666666666666`
- Sobre el conjunto de prueba: `Precisión: 0.9696 Tasa de error: 0.030399999999999983`

Posteriormente se introdujo *early stopping* como mejora para evitar un sobreaprendizaje de la red. Si la red no mejoraba en 15 épocas se detendría el entrenamiento. Se cambió el optimizador *Adam* por *AdamW* que introduce la técnica de *weight decay*. Con estos cambios los resultados eran:
- Sobre el conjunto de entrenamiento: `Precisión: 0.9748833333333333 Tasa de error: 0.025116666666666676`
- Sobre el conjunto de prueba: `Precisión: 0.9638 Tasa de error: 0.03620000000000001`

~~~
#RED 2: Completamente conectada | 3 CAPAS
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', verbose=1, patience=15)
model_red_2 = tf.keras.models.Sequential()
#model_red_2.add(tf.keras.layers.Dense(512, activation='sigmoid'))
model_red_2.add(tf.keras.layers.Dense(256, activation='sigmoid'))
model_red_2.add(tf.keras.layers.Dense(128, activation='sigmoid'))
model_red_2.add(tf.keras.layers.Dense(64, activation='sigmoid'))
model_red_2.add(tf.keras.layers.Dense(10, activation='softmax'))
#Para entrenar el modelo
#tf.keras.optimizers.Adam(learning_rate=0.0015)
#model_red_2.compile('adam', 'categorical_crossentropy') #funcion de perdida
model_red_2.compile('adamW', 'categorical_crossentropy') #funcion de perdida
etiquetas_red_2 = tf.keras.utils.to_categorical(etiquetas_train)
model_red_2.fit(images_train, etiquetas_red_2, epochs=100, callbacks=[early_stopping])
#model_red_2.fit(images_train, etiquetas_red_2, epochs=100)

preds_train_red_2 = model_red_2.predict(images_train).argmax(axis=1)
acc_train = accuracy_score(etiquetas_red_2.argmax(axis=1),preds_train_red_2)
print("Precisión: ", acc_train, "Tasa de error: ", 1 - acc_train)

preds_test_red_2 = model_red_2.predict(images_test).argmax(axis=1)
acc_test = accuracy_score(etiquetas_test, preds_test_red_2)
print("Precisión: ", acc_test, "Tasa de error: ", 1 - acc_test)
~~~

## Tercera red neuronal
En esta red se introdujeron las capas convolucionales. Se sustituyeron dos de las capas completamente conectadas por capas convolucionales. Inicialmente se utilizaron dos capas convolucionales de 16 y 32 neuronas y dos capas de Pooling. Tras estas se usaba una capa completamente conectada de tamaño variable (64, 128, 256 neuronas). También se experimentó con dos capas completamente conectadas en lugar de una.
~~~
#RED 3: CONVOLUCIONAL | 3 CAPAS
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', verbose=1, patience=15)
model_red_3 = tf.keras.models.Sequential()
model_red_3.add(tf.keras.layers.Reshape((28,28,1)))
model_red_3.add(tf.keras.layers.Conv2D(16, 3, activation='relu'))
model_red_3.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model_red_3.add(tf.keras.layers.Conv2D(32, 3, activation='relu'))
model_red_3.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model_red_3.add(tf.keras.layers.Flatten())
#model_red_3.add(tf.keras.layers.Dense(128, activation='sigmoid'))
model_red_3.add(tf.keras.layers.Dense(64, activation='sigmoid'))
model_red_3.add(tf.keras.layers.Dense(10, activation='softmax'))
#Para entrenar el modelo
#tf.keras.optimizers.Adam(learning_rate=0.0015)
model_red_3.compile('adamW', 'categorical_crossentropy') #funcion de perdida
etiquetas_red_3 = tf.keras.utils.to_categorical(etiquetas_train)
model_red_3.fit(images_train, etiquetas_red_3, epochs=100, callbacks=[early_stopping])

preds_train_red_3 = model_red_3.predict(images_train).argmax(axis=1)
acc_train = accuracy_score(etiquetas_red_3.argmax(axis=1),preds_train_red_3)
print("Precisión: ", acc_train, "Tasa de error: ", 1 - acc_train)

preds_test_red_3 = model_red_3.predict(images_test).argmax(axis=1)
acc_test = accuracy_score(etiquetas_test, preds_test_red_3)
print("Precisión: ", acc_test, "Tasa de error: ", 1 - acc_test)
~~~

Con esta red los tiempos de entrenamiento son de 3 minutos aproximadamente y los resultados son:
- Sobre el conjunto de entrenamiento: `Precisión: 0.9816833333333334 Tasa de error: 0.018316666666666648`
- Sobre el conjunto de prueba: `Precisión: 0.9807 Tasa de error: 0.019299999999999984`

Añadiendo una capa totalmente conectada adicional tenemos los siguientes resultados:
- Sobre el conjunto de entrenamiento: `Precisión: 0.9844666666666667 Tasa de error: 0.015533333333333288`
- Sobre el conjunto de prueba: `Precisión: 0.9817 Tasa de error: 0.018299999999999983`

Añadiendo *padding* a las capas convolucionales:
- Sobre el conjunto de entrenamiento: `Precisión: 0.9871833333333333 Tasa de error: 0.012816666666666698`
- Sobre el conjunto de prueba: `Precisión: 0.984 Tasa de error: 0.016000000000000014`

## Forma definitiva
Por último voy a describir la estructura de la red neuronal que mejores resultados a otorgado.
### Incremento en el conjunto de entrenamiento
Uno de los cambios importantes fue la adición a las imágenes de entrenamiento de una versión modificada de esas imágenes. En un principio se introdujeron rotaciones, desplazamientos y zoom pero en el mejor resultado solo se utilizaron rotaciones.
~~~
# INCREMENTO DEL CONJUNTO DE ENTRENAMIENTO
#datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2)
datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=15)
it = datagen.flow(images_train.reshape((-1,28,28,1)), etiquetas_train, batch_size=1)
nuevas_imagenes = [it [i][0] for i in range(60000)]
nuevas_etiquetas = [it [i][1] for i in range(60000)]
fullImagenes= np.concatenate([images_train, np.array(nuevas_imagenes).reshape((60000, 28*28))])
fullEtiquetas= np.concatenate([etiquetas_train, np.array(nuevas_etiquetas).reshape(60000,)])
~~~
### Capas de neuronas
Las distintas pruebas efectuadas orientaron la estructura de la red a la siguiente:
- Capa convolucional de 16 neuronas.
- Capa convolucional de 32 neuronas.
- Capa convolucional de 32 neuronas.
- Capa completamente conectada de 64 neuronas.
- Capa de salida de 10 neuronas.

Más capas desembocaban en peores resultados de forma general y menos neuronas o capas también, lo cual indicaba que este el *sweet spot* en cuanto a capas y tamaño.

### Capas adicionales
Además de las capas de neuronas se añadieron las siguientes capas:
- Capas de *BatchNormalization* después de cada capa convolucional.
- Capas de *Pooling* después de cada capa de BatchNormalization.
- Entre la 2ª y 3ª capa convolucional y entre la 3ª capa convolucional y la capa completamente conectada se han colocado sendas capas de *Dropout*.

### Modificaciones a las capas
Para la función de activación de la capa completamente conectada se ha cambiado de Sigmoide a ReLU. Para las capas convolucionales se ha utilizado un kernel de tamaño 3, del mismo tamaño que el kernel y función de activación ReLU.

### Modificaciones al entrenamiento
El algoritmo de entrenamiento ha sido AdamW que introduce *Weight decay*. La tasa de aprendizaje de partida es de 0.0015 y como valor para el weight decay se ha utilizado 0.005.

Además del *early stopping* se ha introducido una reducción en meseta, de forma que si no se detecta mejora en 4 épocas de disminuye la tasa de aprendizaje en un 25% hasta un mínimo de 0.0002.

Se ha introducido una partición de validación del 20% del conjunto de entrenamiento. Para el early stopping y para la reducción en meseta se monitoriza la pérdida en esta partición de validación en lugar de la pérdida sobre el conjunto de entrenamiento.

Por último, dado que durante los experimentos con estas modificaciones se ha observado que se alcanzaban las 100 épocas que se tenían establecidas de máximo, éste se ha aumentado a 200 épocas.

~~~
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', verbose=1, patience=15)
reduce_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.75, patience=4, min_lr=0.0002, verbose=1)

#Definimos la red
model_definitivo = tf.keras.models.Sequential()
#Añade una capa, las capas estan en keras.layers.Dense
#Este caso es la capa oculta
model_definitivo.add(tf.keras.layers.Reshape((28,28,1)))
model_definitivo.add(tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'))
model_definitivo.add(tf.keras.layers.BatchNormalization())
model_definitivo.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
#model_definitivo.add(tf.keras.layers.Dropout(0.5))
model_definitivo.add(tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'))
model_definitivo.add(tf.keras.layers.BatchNormalization())
model_definitivo.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model_definitivo.add(tf.keras.layers.Dropout(0.5))
model_definitivo.add(tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'))
model_definitivo.add(tf.keras.layers.BatchNormalization())
model_definitivo.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model_definitivo.add(tf.keras.layers.Flatten())
model_definitivo.add(tf.keras.layers.Dropout(0.5))
model_definitivo.add(tf.keras.layers.Dense(64, activation='relu'))
#model.add(tf.keras.layers.Dropout(0.2))
#Capa de salida
model_definitivo.add(tf.keras.layers.Dense(10, activation='softmax'))
#Para entrenar el modelo
optimizador=tf.keras.optimizers.AdamW(learning_rate=0.0015, weight_decay=0.005)
model_definitivo.compile(optimizer=optimizador, loss='categorical_crossentropy', metrics=['accuracy']) #funcion de perdida
etiquetas_definitivas = tf.keras.utils.to_categorical(fullEtiquetas)
model_definitivo.fit(fullImagenes, etiquetas_definitivas, epochs=200, callbacks=[early_stopping, reduce_on_plateau], validation_split=0.2) #Se le añade el callback personalizado
~~~

El tiempo de entrenamiento se ha incrementado de forma considerable, llegando a casi 15 minutos. Los mejores resultados que se han obtenido son los siguientes:
- Sobre el conjunto de entrenamiento: `Precisión: 0.9981166666666667 Tasa de error: 0.001883333333333348`
- Sobre el conjunto de prueba: `Precisión: 0.9964 Tasa de error: 0.0036000000000000476`

Esta red neuronal más compleja y con múltiples técnicas de apoyo tiene unos resultados buenos, cerca del 0.3% de error que se tiene como objetivo máximo.

## Sobre los resultados y conclusiones
Aunque se tiene claro por los experimentos realizados que se ha conseguido un número de capas de neuronas óptimo, los resultados entre cada entrenamiento son muy variables. Por desgracia no puedo aseverar las razones de esto.

No se tiene certeza sobre si la posición de las capas convolucionales podría tener unos mejores resultados. La experimentación llevada a cabo no muestra signos de ello.

Hay que señalar también que ha sido una ventaja contar con una GPU *modesta* a la hora de realizar la práctica ya que la ejecución en Colab es bastante lenta.

Por último, sería interesante que, una vez terminada la práctica (y el tiempo de entrega tardío), se proporcionarse la mejor solución que se tenga del problema realizada por personas con muchos más conocimientos que nosotros.