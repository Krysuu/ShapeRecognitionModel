import glob
import sys
import argparse
from timeit import default_timer as timer

import tensorflow.keras.applications.xception as xception
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD

from save_util import *


class TimingCallback(Callback):
    def __init__(self, logs={}):
        super().__init__()
        self.logs = []

    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(timer() - self.starttime)

    def clear(self):
        self.logs = []


time_cb = TimingCallback()


def prepare_result_directory(result_directory):
    isExist = os.path.exists(result_directory)
    if not isExist:
        os.makedirs(result_directory)


def load_data_to_dataframe(dataset_dir: str):
    image_paths = []
    image_labels = []
    for filename in glob.glob(dataset_dir + '/*/*.png'):
        image_paths.append(filename)
        image_label = filename.split('\\')[1]
        image_labels.append(image_label)

    return pd.DataFrame(list(zip(image_paths, image_labels)), columns=['filename', 'label'])


def get_dataset_partitions_pd(df, train_split=0.7, val_split=0.15, test_split=0.15, target_variable=None,
                              random_state=12):
    assert (train_split + test_split + val_split) == 1
    assert val_split == test_split

    df_sample = df.sample(frac=1, random_state=random_state)

    if target_variable is not None:
        grouped_df = df_sample.groupby(target_variable)
        arr_list = [np.split(g, [int(train_split * len(g)), int((1 - val_split) * len(g))]) for i, g in grouped_df]

        train_ds = pd.concat([t[0] for t in arr_list])
        val_ds = pd.concat([t[1] for t in arr_list])
        test_ds = pd.concat([v[2] for v in arr_list])

    else:
        indices_or_sections = [int(train_split * len(df)), int((1 - val_split) * len(df))]
        train_ds, val_ds, test_ds = np.split(df_sample, indices_or_sections)

    return train_ds, val_ds, test_ds


def preprocess_and_load_data(train_dataframe, validation_dataframe, test_dataframe, preprocessing_function, batch_size):
    train_idg = ImageDataGenerator(preprocessing_function=preprocessing_function)
    validation_idg = ImageDataGenerator(preprocessing_function=preprocessing_function)
    test_idg = ImageDataGenerator(preprocessing_function=preprocessing_function)

    train_data = train_idg.flow_from_dataframe(
        train_dataframe,
        x_col='filename',
        y_col='label',
        target_size=(ROWS, COLS),
        batch_size=batch_size
    )

    validation_data = validation_idg.flow_from_dataframe(
        validation_dataframe,
        x_col='filename',
        y_col='label',
        target_size=(ROWS, COLS),
        batch_size=1,
        shuffle=False
    )

    test_data = test_idg.flow_from_dataframe(
        test_dataframe,
        x_col='filename',
        y_col='label',
        target_size=(ROWS, COLS),
        batch_size=1,
        shuffle=False
    )
    return train_data, validation_data, test_data


def prepare_model(pretrained_model, train_gen, learning_rate, momentum, dropout, dense_size):
    nclass = len(train_gen.class_indices)
    model = Sequential()
    model.add(pretrained_model)
    model.add(Flatten())
    model.add(Dense(dense_size, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(nclass, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(learning_rate=learning_rate, momentum=momentum),
                  metrics=['accuracy'])
    model.summary()
    return model


def fit_model(model, train_gen, validation_gen, weights_path, epochs):
    checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early = EarlyStopping(monitor='val_loss', patience=5)

    callbacks_list = [checkpoint, early, time_cb]
    return model.fit(train_gen,
                     epochs=epochs,
                     shuffle=True,
                     verbose=2,
                     callbacks=[callbacks_list],
                     validation_data=validation_gen
                     )


def perform_test(preprocess_input, pretrained_model, n_tests, data_directory, result_directory, dense_size, batch_size,
                 learning_rate, momentum):
    df = load_data_to_dataframe(data_directory)
    prepare_result_directory(result_directory)

    for i in range(n_tests):
        print(f'Current fold: {i + 1}/{n_tests}')
        train_ds, val_ds, test_ds = get_dataset_partitions_pd(df, target_variable='label', random_state=i)

        train_gen, validation_gen, test_gen = preprocess_and_load_data(train_dataframe=train_ds,
                                                                       validation_dataframe=val_ds,
                                                                       test_dataframe=test_ds,
                                                                       preprocessing_function=preprocess_input,
                                                                       batch_size=batch_size
                                                                       )

        model = prepare_model(pretrained_model,
                              train_gen,
                              learning_rate,
                              momentum,
                              dropout,
                              dense_size
                              )

        history = fit_model(model=model,
                            train_gen=train_gen,
                            validation_gen=validation_gen,
                            weights_path=weights_path,
                            epochs=epochs,
                            )

        save_training_progress(history=history, current_fold=i, result_directory=result_directory)

        model.load_weights(weights_path)
        predicts = model.predict(test_gen, verbose=True)

        save_test_results(test_gen=test_gen, predicts=predicts, result_directory=result_directory,
                          current_test=i)
        save_misclassified(test_gen=test_gen,
                           predicts=predicts,
                           result_directory=result_directory,
                           current_fold=i
                           )

        save_and_reset_time_logs(result_directory, time_cb)


# Constant
ROWS = 299
COLS = 299
n_tests = 5
weights_path = "weights.best.hdf5"
epochs = 50
dropout = 0.5

parser = argparse.ArgumentParser()
parser.add_argument('dataset_path', type=str, help='path to dataset directory')
parser.add_argument('result_path', type=str, help='path to result directory')
parser.add_argument('--dense_size', type=int, default=512, help='Size of final dense layer')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning_rate')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
# parser.add_argument('--decay_rate', type=float, default=0.00001, help='Decay rate')
args = vars(parser.parse_args())

pretrained_model = xception.Xception(weights="imagenet",
                                     include_top=False,
                                     input_shape=(ROWS, COLS, 3)
                                     )
pretrained_model.trainable = False

perform_test(xception.preprocess_input,
             pretrained_model,
             n_tests,
             args['dataset_path'],
             args['result_path'],
             args['dense_size'],
             args['batch_size'],
             args['learning_rate'],
             args['momentum']
             )
