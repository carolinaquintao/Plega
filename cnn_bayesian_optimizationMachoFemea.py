from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Conv2D, Dropout, BatchNormalization, MaxPooling2D, Flatten, Activation
from keras.datasets import mnist
from keras.utils import np_utils
import tensorflow as tf
from keras import backend as K
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.model_selection import train_test_split
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
40095179
guia 98119310
98119311

NUM_CLASSES = 2

def get_input_datasets(use_bfloat16=False):
    """Downloads the MNIST dataset and creates train and eval dataset objects.

    Args:
      use_bfloat16: Boolean to determine if input should be cast to bfloat16

    Returns:
      Train dataset, eval dataset and input shape.

    """
    # Caminho do diretório onde estão os dados
    data_dir = "C:/Users/carol/OneDrive/Projeto Plega/Base de dados/BaseMF/"
    
    # Inicializar listas para armazenar imagens e rótulos
    images = []
    labels = []
    img_rows, img_cols = 64, 128
    cast_dtype = tf.bfloat16 if use_bfloat16 else tf.float32

    # Dicionário de mapeamento de classes (assumindo que as subpastas representam classes)
    class_names = os.listdir(data_dir)
    class_dict = {class_name: idx for idx, class_name in enumerate(class_names)}
    
    # Carregar as imagens e converter em tensores
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            # Carregar a imagem e redimensionar
            img = load_img(img_path, target_size=(img_rows, img_cols), color_mode='grayscale')
            # Converter para array
            img_array = img_to_array(img)
            # Normalizar os valores da imagem
            img_array /= 255.0
            # Armazenar a imagem e o rótulo correspondente
            images.append(img_array)
            labels.append(class_dict[class_name])
    
    # Converter listas para arrays numpy
    images = np.array(images)
    labels = np.array(labels)
    
    # Dividir em conjunto de treino e teste
    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    
    # Converter rótulos em categóricos (one-hot encoding)
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=len(class_names))
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=len(class_names))
    
    # train dataset
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.repeat()
    train_ds = train_ds.map(lambda x, y: (tf.cast(x, cast_dtype), y))
    train_ds = train_ds.batch(64, drop_remainder=True)

    # eval dataset
    eval_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    eval_ds = eval_ds.repeat()
    eval_ds = eval_ds.map(lambda x, y: (tf.cast(x, cast_dtype), y))
    eval_ds = eval_ds.batch(64, drop_remainder=True)

    input_shape = (img_rows, img_cols, 1)

    return train_ds, eval_ds, input_shape

def get_model(input_shape, dropout2_rate=0.5):

    # input image dimensions
    # img_rows, img_cols = 512, 1024


    # Define a CNN model to recognize MNIST.
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape, name="conv2d_1"))
    model.add(Conv2D(64, (3, 3), activation='relu', name="conv2d_2"))
    model.add(MaxPooling2D(pool_size=(2, 2), name="maxpool2d_1"))
    model.add(Dropout(0.25, name="dropout_1"))
    model.add(Flatten(name="flatten"))
    model.add(Dense(128, activation='relu', name="dense_1"))
    model.add(Dropout(dropout2_rate, name="dropout_2"))
    model.add(Dense(NUM_CLASSES, activation='softmax', name="dense_2"))

    return model

train_ds, eval_ds, input_shape = get_input_datasets()

def fit_with(input_shape, verbose, dropout2_rate, lr):

    # Create the model using a specified hyperparameters.
    model = get_model(input_shape, dropout2_rate)

    # Train the model for a specified number of epochs.
    optimizer= Adam(learning_rate = lr)

    model.compile(optimizer=optimizer,
                  loss    = 'mse',
                  metrics = ['accuracy'])

    # Train the model with the train dataset.
    model.fit(train_ds,
              validation_data  = eval_ds,
              epochs           = 5,
              validation_steps = 60000 // 32,
              steps_per_epoch  = 60000 // 32,
              verbose          = verbose)

    # Evaluate the model with the eval dataset.
    score = model.evaluate(eval_ds, steps = 5, verbose=1)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print('\n')

    # Return the accuracy.

    return score[1]

from functools import partial

input_shape = input_shape
verbose     = 1

fit_with_partial = partial(fit_with, input_shape, verbose)

"""The BayesianOptimization object will work out of the box without much tuning needed. The main method you should be aware of is maximize, which does exactly what you think it does.

There are many parameters you can pass to maximize, nonetheless, the most important ones are:

n_iter: How many steps of bayesian optimization you want to perform. The more steps the more likely to find a good maximum you are.
init_points: How many steps of random exploration you want to perform. Random exploration can help by diversifying the exploration space.
"""

# Bounded region of parameter space

bounds = {'lr'           :(1e-4, 1e-2),
          'dropout2_rate':(0.05, 0.5),
          'batch_size'   :(1, 4.001),
          'num_filters'  :(1, 4.001),
          'kernel_size'  :(2, 4.001)}


bounds_2 = {'dropout2_rate': (0.1, 0.5),
            'lr'           : (1e-4, 1e-2)}

from bayes_opt import BayesianOptimization


optimizer = BayesianOptimization(
    f            = fit_with_partial,
    pbounds      = bounds_2,
    verbose      = 1,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state = 1
)

from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
logger = JSONLogger(path="./logs.log")
optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

optimizer.maximize(init_points = 5, n_iter = 2,)

for i, res in enumerate(optimizer.res):
    print("Iteration {}: \n\t{}".format(i, res))

print(optimizer.max)

from bayes_opt.util import load_logs


new_optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds={"x": (-2, 2), "y": (-2, 2)},
    verbose=2,
    random_state=7,
)

# New optimizer is loaded with previously seen points
load_logs(new_optimizer, logs=["./logs.log"]);

"""# References

- https://github.com/fmfn/BayesianOptimization
- https://stackoverflow.com/questions/55586472/mnist-data-set-up-batch
- https://keras.io/examples/mnist_cnn/
- https://www.youtube.com/watch?v=sXdxyUCCm8s
- https://machinelearningapplied.com/hyperparameter-search-with-bayesian-optimization-for-keras-cnn-classification-and-ensembling/
- https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam
- https://stackoverflow.com/questions/55586472/mnist-data-set-up-batch
"""

