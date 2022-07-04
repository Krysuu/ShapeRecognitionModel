import argparse

import tensorflow.keras.applications.xception as xception
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD

from load_save_util import *


def scheduler(epoch, lr):
    if epoch < 1:
        return lr
    else:
        return lr * decay_rate


def start(data_directory, dense_size, batch_size, learning_rate, momentum, binary_class, is_binary):
    df = load_data_to_dataframe(data_directory, binary_class, is_binary)
    train_idg = ImageDataGenerator(preprocessing_function=xception.preprocess_input)

    class_mode = 'categorical'
    if is_binary:
        class_mode = 'binary'

    train_data = train_idg.flow_from_dataframe(
        df,
        x_col='filename',
        y_col='label',
        class_mode=class_mode,
        target_size=(ROWS, COLS),
        batch_size=batch_size
    )

    pretrained_model = xception.Xception(weights="imagenet",
                                         include_top=False,
                                         input_shape=(ROWS, COLS, 3)
                                         )
    pretrained_model.trainable = False

    model = Sequential()
    model.add(pretrained_model)
    model.add(Flatten())
    model.add(Dense(dense_size, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(dense_size, activation='relu'))
    model.add(Dropout(dropout))

    if is_binary:
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer=SGD(learning_rate=learning_rate, momentum=momentum),
                      metrics=['accuracy'])
    else:
        model.add(Dense(len(train_idg.class_indices), activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer=SGD(learning_rate=learning_rate, momentum=momentum),
                      metrics=['accuracy'])

    model.summary()

    checkpoint = ModelCheckpoint(weights_path, monitor='loss', verbose=1, save_best_only=True, mode='min')
    early = EarlyStopping(monitor='loss', patience=3)
    lr_callback = LearningRateScheduler(scheduler, verbose=0)

    callbacks_list = [checkpoint, early, lr_callback]
    model.fit(train_data,
              epochs=epochs,
              shuffle=True,
              verbose=2,
              callbacks=[callbacks_list],
              workers=4
              )

    model.save('model')


# Constant
ROWS = 299
COLS = 299
weights_path = "weights.best.hdf5"
epochs = 50
dropout = 0.5

parser = argparse.ArgumentParser()
parser.add_argument('dataset_path', type=str, help='path to dataset directory')
parser.add_argument('--dense_size', type=int, default=256, help='Size of final dense layer')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning_rate')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--decay_rate', type=float, default=0.94, help='Decay rate')
parser.add_argument('--binary_class', type=str, default="", help='Binary class')
args = vars(parser.parse_args())

decay_rate = args['decay_rate']
binary_class = args['binary_class']
is_binary = False
if binary_class != "":
    is_binary = True

start(args['dataset_path'],
      args['dense_size'],
      args['batch_size'],
      args['learning_rate'],
      args['momentum'],
      binary_class,
      is_binary)
