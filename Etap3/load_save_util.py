import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras.callbacks import Callback
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import glob
from timeit import default_timer as timer
import math


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


def prepare_result_directory(result_directory):
    isExist = os.path.exists(result_directory)
    if not isExist:
        os.makedirs(result_directory)


def load_data_to_dataframe(dataset_dir, binary_class, is_binary):
    image_paths = []
    image_labels = []
    for filename in glob.glob(dataset_dir + '/*/*.*'):
        image_paths.append(filename)
        image_label = filename.split('\\')[1]
        image_labels.append(image_label)

    df = pd.DataFrame(list(zip(image_paths, image_labels)), columns=['filename', 'label'])

    if is_binary:
        nclass = df['label'].nunique()
        class_df = df.loc[df['label'] == binary_class]
        n_images = len(class_df.index)
        other_df = df.loc[df['label'] != binary_class]
        other_df = other_df.groupby('label').head(math.floor(n_images / (nclass - 1)))
        other_df = other_df.assign(label='inny')
        df = pd.concat([class_df, other_df])

    return df


def save_training_progress(history, current_fold, result_directory):
    plt.clf()
    plt.plot(history.history['accuracy'])
    plt.grid()
    plt.title('Trafność na zbiorze treningowym w czasie')
    plt.ylabel('Trafność')
    plt.xlabel('Liczba epok')
    plt.savefig(f'{result_directory}/{current_fold}_przebieg_treningu-trafnosc.png')

    plt.clf()
    plt.plot(history.history['loss'])
    plt.grid()
    plt.title('Funkcja straty na zbiorze treningowym w czasie')
    plt.ylabel('Wartość funkcji straty')
    plt.xlabel('Liczba epok')
    plt.savefig(f'{result_directory}/{current_fold}_przebieg_treningu-funkcja-straty.png')

    csv_path = f'{result_directory}/przebieg_treningu-trafnosc.csv'
    save_or_append_csv(csv_path, history.history['accuracy'])

    csv_path = f'{result_directory}/przebieg_treningu-funkcja-straty.csv'
    save_or_append_csv(csv_path, history.history['loss'])


def save_or_append_csv(csv_path, values):
    if os.path.exists(csv_path):
        append_write = 'a'  # append if already exists
    else:
        append_write = 'w'  # make a new file if not
    with open(csv_path, append_write) as f:
        write = csv.writer(f)
        write.writerow(values)


def save_test_results(test_gen, predicts, result_directory, current_test, is_binary):
    plt.clf()
    plt.figure(figsize=(10, 10))
    label_names = list(test_gen.class_indices)
    y_true = test_gen.labels

    if is_binary:
        predicts_flat = predicts.flatten()
        y_pred = np.rint(predicts_flat)
    else:
        y_pred = np.argmax(predicts, axis=1)

    cm = confusion_matrix(y_true, y_pred)

    ax = plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', ax=ax)

    ax.set_xlabel('Klasa predykowana')
    ax.set_ylabel('Klasa rzeczywista')
    ax.set_title('Macierz pomyłek')
    ax.set_xticklabels(label_names, rotation=45)
    ax.set_yticklabels(label_names, rotation=45)
    plt.savefig(f'{result_directory}/{current_test}_macierz_pomylek.png')

    test_acc = accuracy_score(y_true, y_pred)
    csv_path = f'{result_directory}/trafnosc_test.csv'
    if os.path.exists(csv_path):
        append_write = 'a'  # append if already exists
    else:
        append_write = 'w'  # make a new file if not
    with open(csv_path, append_write) as f:
        write = csv.writer(f)
        write.writerow([test_acc])

    print(f'Test {current_test} accuracy: {test_acc}')


def save_misclassified(test_gen, predicts, result_directory, current_fold, is_binary, binary_class):
    if is_binary:
        save_misclassified_binary(test_gen, predicts, result_directory, current_fold, binary_class)
    else:
        save_misclassified_standard(test_gen, predicts, result_directory, current_fold)


def save_misclassified_standard(test_gen, predicts, result_directory, current_fold):
    label_index = {v: k for k, v in test_gen.class_indices.items()}
    columns = list(label_index.values())

    predicts_df = pd.DataFrame(predicts, columns=columns)

    predicts_top1 = np.argmax(predicts, axis=1)
    predicts_top1 = [label_index[p] for p in predicts_top1]

    misclassified_df = pd.DataFrame(columns=['Nazwa pliku', 'Klasa rzeczywista', 'Klasa predykowana'])
    misclassified_df['Nazwa pliku'] = [os.path.basename(x) for x in test_gen.filenames]
    misclassified_df['Klasa rzeczywista'] = [label_index[p] for p in test_gen.labels]
    misclassified_df['Klasa predykowana'] = predicts_top1

    misclassified_df = pd.concat([misclassified_df, predicts_df], axis=1)
    misclassified_df = misclassified_df.iloc[
        np.where((misclassified_df['Klasa rzeczywista'] != misclassified_df['Klasa predykowana']))]

    csv_path = f'{result_directory}/{current_fold}_blednie_zaklasyfikowane.csv'
    misclassified_df.to_csv(csv_path, index=False)


def save_misclassified_binary(test_gen, predicts, result_directory, current_fold, binary_class):
    label_index = {v: k for k, v in test_gen.class_indices.items()}
    predicts_flat = predicts.flatten()
    predicts_0_1 = np.rint(predicts_flat)
    predicts_labels = [label_index[p] for p in predicts_0_1]

    misclassified_df = pd.DataFrame(
        columns=['Nazwa pliku', 'Klasa rzeczywista', 'Klasa predykowana', 'Prawdopodobieństwo'])
    misclassified_df['Nazwa pliku'] = [os.path.basename(x) for x in test_gen.filenames]
    misclassified_df['Klasa rzeczywista'] = [label_index[p] for p in test_gen.labels]
    misclassified_df['Klasa predykowana'] = predicts_labels
    misclassified_df['Prawdopodobieństwo'] = predicts_flat

    # zamiana przedziału 0-1 na prawdopodobieństwo klasy predykowanej
    misclassified_df['Prawdopodobieństwo'] = np.where(misclassified_df['Klasa predykowana'] == binary_class, 1, 0) + \
                                             misclassified_df['Prawdopodobieństwo'] * \
                                             np.where(misclassified_df['Klasa predykowana'] == binary_class, -1, 1)

    misclassified_df = misclassified_df.iloc[
        np.where((misclassified_df['Klasa rzeczywista'] != misclassified_df['Klasa predykowana']))]

    csv_path = f'{result_directory}/{current_fold}_blednie_zaklasyfikowane.csv'
    misclassified_df.to_csv(csv_path, index=False)


def save_and_reset_time_logs(result_directory, time_cb):
    csv_path = f'{result_directory}/czas_epoka.csv'
    if os.path.exists(csv_path):
        append_write = 'a'  # append if already exists
    else:
        append_write = 'w'  # make a new file if not
    with open(csv_path, append_write) as f:
        write = csv.writer(f)
        write.writerow(time_cb.logs)

    csv_path = f'{result_directory}/czas_suma.csv'
    if os.path.exists(csv_path):
        append_write = 'a'  # append if already exists
    else:
        append_write = 'w'  # make a new file if not
    with open(csv_path, append_write) as f:
        write = csv.writer(f)
        write.writerow([sum(time_cb.logs)])

    time_cb.clear()
