import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


def save_training_progress(history, current_fold, result_directory):
    plt.clf()
    plt.plot(history.history['val_accuracy'])
    plt.grid()
    plt.title('Trafność na zbiorze walidacyjnym w czasie')
    plt.ylabel('Trafność')
    plt.xlabel('Liczba epok')
    plt.savefig(f'{result_directory}/{current_fold}_przebieg_treningu-trafnosc.png')

    plt.clf()
    plt.plot(history.history['val_loss'])
    plt.grid()
    plt.title('Funkcja straty na zbiorze walidacyjnym w czasie')
    plt.ylabel('Wartość funkcji straty')
    plt.xlabel('Liczba epok')
    plt.savefig(f'{result_directory}/{current_fold}_przebieg_treningu-funkcja-straty.png')

    csv_path = f'{result_directory}/przebieg_treningu-trafnosc.csv'
    save_or_append_csv(csv_path, history.history['val_accuracy'])

    csv_path = f'{result_directory}/przebieg_treningu-funkcja-straty.csv'
    save_or_append_csv(csv_path, history.history['val_loss'])


def save_or_append_csv(csv_path, values):
    if os.path.exists(csv_path):
        append_write = 'a'  # append if already exists
    else:
        append_write = 'w'  # make a new file if not
    with open(csv_path, append_write) as f:
        write = csv.writer(f)
        write.writerow(values)


def save_test_results(test_gen, predicts, result_directory, current_test):
    plt.clf()
    label_names = list(test_gen.class_indices)
    y_true = test_gen.labels
    y_pred = np.argmax(predicts, axis=1)

    cm = confusion_matrix(y_true, y_pred)

    ax = plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', ax=ax)

    ax.set_xlabel('Klasa predykowana')
    ax.set_ylabel('Klasa rzeczywista')
    ax.set_title('Macierz pomyłek')
    ax.xaxis.set_ticklabels(label_names)
    ax.yaxis.set_ticklabels(label_names)
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


def save_misclassified(test_gen, predicts, result_directory, current_fold):
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
