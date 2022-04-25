import csv
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dropout, Flatten, Dense, AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.optimizers import SGD

import tensorflow.keras.applications.resnet50 as resnet50
import tensorflow.keras.applications.xception as xception
import tensorflow.keras.applications.vgg16 as vgg16


def load_data_to_dataframe(dataset_dir: str):
    image_paths = []
    image_labels = []
    for filename in glob.glob(dataset_dir + '/*/*.png'):
        image_paths.append(filename)
        image_label = filename.split('\\')[1]
        image_labels.append(image_label)

    return pd.DataFrame(list(zip(image_paths, image_labels)), columns=['filename', 'label'])


def preprocess_and_load_data(train_dataframe, test_dataframe, preprocessing_function, batch_size):
    train_idg = ImageDataGenerator(preprocessing_function=preprocessing_function)
    test_idg = ImageDataGenerator(preprocessing_function=preprocessing_function)

    train_data = train_idg.flow_from_dataframe(
        train_dataframe,
        x_col='filename',
        y_col='label',
        target_size=(ROWS, COLS),
        batch_size=batch_size
    )

    test_data = test_idg.flow_from_dataframe(
        test_dataframe,
        x_col='filename',
        y_col='label',
        target_size=(ROWS, COLS),
        batch_size=1,
        shuffle=False
    )
    return train_data, test_data


def prepare_model(pretrained_model, train_gen, learning_rate, dropout, dense_size):
    nclass = len(train_gen.class_indices)
    model = Sequential()
    model.add(pretrained_model)
    model.add(Flatten())
    model.add(Dense(dense_size, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(nclass, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(learning_rate=learning_rate),
                  metrics=['accuracy'])
    model.summary()
    return model


def fit_model(model, train_gen, weights_path, epochs):
    checkpoint = ModelCheckpoint(weights_path, monitor='accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    return model.fit(train_gen,
                     epochs=epochs,
                     shuffle=True,
                     verbose=2,
                     callbacks=[callbacks_list]
                     )


def save_training_progress(history, current_fold, result_directory):
    plt.clf()
    plt.plot(history.history['accuracy'])
    plt.grid()
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['accuracy'])
    plt.savefig(f'{result_directory}/fold_{current_fold}_training_progress.png')

    csv_path = f'{result_directory}/training_history.csv'
    if os.path.exists(csv_path):
        append_write = 'a'  # append if already exists
    else:
        append_write = 'w'  # make a new file if not
    with open(csv_path, append_write) as f:
        write = csv.writer(f)
        write.writerow(history.history['accuracy'])


def save_fold_test(test_gen, predicts, result_directory, current_fold):
    plt.clf()
    label_names = list(test_gen.class_indices)
    y_true = test_gen.labels
    y_pred = np.argmax(predicts, axis=1)

    cm = confusion_matrix(y_true, y_pred)

    ax = plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', ax=ax)

    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(label_names)
    ax.yaxis.set_ticklabels(label_names)
    plt.savefig(f'{result_directory}/{current_fold}_fold_confusion_matrix.png')

    test_acc = accuracy_score(y_true, y_pred)
    csv_path = f'{result_directory}/test_acc.csv'
    if os.path.exists(csv_path):
        append_write = 'a'  # append if already exists
    else:
        append_write = 'w'  # make a new file if not
    with open(csv_path, append_write) as f:
        write = csv.writer(f)
        write.writerow([test_acc])

    print(f'Fold {current_fold} accuracy: {test_acc}')


def save_misclassified(test_gen, predicts, result_directory, current_fold):
    label_index = {v: k for k, v in test_gen.class_indices.items()}
    columns = list(label_index.values())

    predicts_df = pd.DataFrame(predicts, columns=columns)

    predicts_top1 = np.argmax(predicts, axis=1)
    predicts_top1 = [label_index[p] for p in predicts_top1]

    misclassified_df = pd.DataFrame(columns=['fname', 'true_label', 'predicted_label'])
    misclassified_df['fname'] = [os.path.basename(x) for x in test_gen.filenames]
    misclassified_df['true_label'] = [label_index[p] for p in test_gen.labels]
    misclassified_df['predicted_label'] = predicts_top1

    misclassified_df = pd.concat([misclassified_df, predicts_df], axis=1)
    misclassified_df = misclassified_df.iloc[
        np.where((misclassified_df['true_label'] != misclassified_df['predicted_label']))]

    csv_path = f'{result_directory}/{current_fold}_fold_misclassified.csv'
    misclassified_df.to_csv(csv_path, index=False)


def prepare_result_directory(result_directory):
    isExist = os.path.exists(result_directory)
    if not isExist:
        os.makedirs(result_directory)


def perform_test(preprocess_input, pretrained_model):
    df = load_data_to_dataframe(data_directory)
    prepare_result_directory(result_directory)

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)
    current_fold = 0
    for fold in kf.split(df, df['label']):
        current_fold += 1
        if skip_folds_after != 0 and current_fold > skip_folds_after:
            break

        print(f'Current fold: {current_fold}/{n_splits}')
        train_df = df.iloc[fold[0]]
        test_df = df.iloc[fold[1]]

        train_gen, test_gen = preprocess_and_load_data(train_dataframe=train_df,
                                                       test_dataframe=test_df,
                                                       preprocessing_function=preprocess_input,
                                                       batch_size=batch_size
                                                       )

        model = prepare_model(pretrained_model=pretrained_model,
                              train_gen=train_gen,
                              learning_rate=learning_rate,
                              dropout=dropout,
                              dense_size=dense_size
                              )
        history = fit_model(model=model, train_gen=train_gen, weights_path=weights_path, epochs=epochs)

        save_training_progress(history=history, current_fold=current_fold, result_directory=result_directory)

        model.load_weights(weights_path)
        predicts = model.predict(test_gen, verbose=True, workers=2)

        save_fold_test(test_gen=test_gen, predicts=predicts, result_directory=result_directory,
                       current_fold=current_fold)
        save_misclassified(test_gen=test_gen,
                           predicts=predicts,
                           result_directory=result_directory,
                           current_fold=current_fold
                           )


# Constant
# ROWS = 500
# COLS = 500
n_splits = 5
weights_path = "weights.best.hdf5"
batch_size = 64
epochs = 30
learning_rate = 0.01
dropout = 0.5
dense_size = 64
skip_folds_after = 0  # 0 to run all

# Main
if len(sys.argv) != 5:
    print("Invalid parameters")
    exit()

# Variables
# Lokalizacja danych wejściowych np. white_background_data
data_directory = sys.argv[1].lower()

# Lokalizacja danych wyjściowych np. results/white_background_data
result_directory = sys.argv[2].lower()

# ROWS: xception, resnet50, vgg16
ROWS = int(sys.argv[3])

# COLS:
COLS = int(sys.argv[4])

pretrained_model = xception.Xception(weights="imagenet",
                                     include_top=False,
                                     input_shape=(ROWS, COLS, 3)
                                     )
pretrained_model.trainable = False

perform_test(preprocess_input=xception.preprocess_input, pretrained_model=pretrained_model)
