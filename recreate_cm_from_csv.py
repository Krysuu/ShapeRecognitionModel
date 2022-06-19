import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


def save_test_results(cm, n):
    plt.clf()
    plt.figure(figsize=(10, 10))
    label_names = ["ceownik", "dwuteownik", "kątownik", "kwadratowy", "okrągły", "płaskownik", "profil", "rura"]

    ax = plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', ax=ax)

    ax.set_xlabel('Klasa predykowana')
    ax.set_ylabel('Klasa rzeczywista')
    ax.set_title('Macierz pomyłek')
    ax.set_xticklabels(label_names, rotation=45)
    ax.set_yticklabels(label_names, rotation=45)
    plt.savefig(f'{n}_macierz_pomylek.png')


def create(n):
    max_elements = 200
    loc = f'{n}_blednie_zaklasyfikowane.csv'
    df = pd.read_csv(loc)
    label_names = ["ceownik", "dwuteownik", "katownik", "kwadratowy", "okragly", "plaskownik", "profil", "rura"]

    cm = np.zeros(shape=(8, 8), dtype=int)

    i = 0
    for label1 in label_names:
        other_labels = label_names.copy()
        other_labels.remove(label1)
        j = 0
        for label2 in label_names:
            count = df[(df['Klasa rzeczywista'] == label1) & (df['Klasa predykowana'] == label2)].shape[0]
            cm[i, j] = count
            j += 1

        cm[i, i] = max_elements - sum(cm[i])
        i += 1

    save_test_results(cm, n)


for n in range(1, 6):
    create(n)
