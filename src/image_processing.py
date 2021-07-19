import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
from tqdm.notebook import tqdm
from shutil import copyfile, rmtree
from pathlib import Path
from threading import Thread, Semaphore, Lock
from collections import deque
from itertools import chain
import datasets as ds


def load_show_img(path):
    img = cv2.imread(path)
    plt.imshow(img)
    plt.show()


def show_img(img):
    plt.imshow(img)
    plt.show()


def show_img_gray(img):
    plt.imshow(img, cmap="gray", vmin=0, vmax=255)
    plt.show()


def load_img_rgb(path):
    return cv2.imread(path, cv2.IMREAD_COLOR)


def load_img_gray(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def load_img(paths, img_queue, sem):

    for path in paths:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img_queue.append(img)
    sem.release()


def get_all_img_parallel(paths, num_workers=16, chunk_size=16):

    img_queue = deque()
    thread_sem = Semaphore(num_workers)

    i = 0

    tasks = {i: [] for i in range(num_workers)}

    for i, path in enumerate(paths):

        tasks[i % num_workers].append(path)
        # print(chunk_size*num_workers - i % (chunk_size * num_workers), num_workers - i % num_workers)
        if (
            chunk_size * num_workers - i % (chunk_size * num_workers)
            == num_workers - i % num_workers
        ):
            thread_sem.acquire()
            Thread(
                target=load_img, args=(tasks[i % num_workers], img_queue, thread_sem)
            ).start()
            tasks[i % num_workers] = []

        if i % chunk_size == 0:
            while True:
                try:
                    yield img_queue.popleft()
                except IndexError:
                    break

    thread_sem.acquire()
    load_img(chain.from_iterable(tasks.values()), img_queue, thread_sem)
    while True:
        try:
            yield img_queue.popleft()
        except IndexError:
            break


def dir_get_all_img(path, formats=["jpg", "jpeg", "png"]):

    if os.path.isdir(path):
        for content in os.listdir(path):
            for img in dir_get_all_img(os.path.join(path, content), formats):
                yield img

    for f in formats:
        if path.endswith(f):
            yield path

    return


def dir_get_all_frontal(path, formats=["jpg", "jpeg", "png"]):

    if os.path.isdir(path):
        for content in os.listdir(path):
            for img in dir_get_all_frontal(os.path.join(path, content), formats):
                yield img

    for f in formats:
        if path.endswith("frontal." + f):
            yield path

    return


def shrink_by_factor(img, factor):
    return cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)


def shrink_width_keep_aspect(img, new_width=320):

    height, width = img.shape[:2]
    factor = float(new_width) / width
    new_height = int(np.round(height * factor))
    return cv2.resize(img, (new_height, new_width), interpolation=cv2.INTER_AREA)


def subsample_pathologies(labels_path, per_class = 20000, val_split = 0.1, no_finding_lim=7810, patho = ['Atelectasis', 'Lung Opacity', 'Pleural Effusion', 'Support Devices', ]):
    
    """
        Subsamples dataset to specifiec pathologies and no-finding. Excludes uncertain diagnosis
    """

    if no_finding_lim > per_class:
        no_finding_lim = per_class

    labels = pd.read_csv(labels_path)

    patho_labels = labels[patho].values

    uncertain = (patho_labels == 1).any(axis=1)
    present = (patho_labels == 2).any(axis=1)
    frontal = labels["Frontal/Lateral"] == "Frontal"
    no_finding = labels["No Finding"] == 1

    # image shapes
    wide_angel = (labels["height"] / labels["width"]) <= 1
    not_too_wide = (labels["height"] / labels["width"]) >= 3 / 4

    fine_sizes = np.logical_and(wide_angel, not_too_wide)

    # print(no_finding.sum(), np.logical_and(fine_sizes, no_finding).sum())

    bool_indices = np.logical_or(present, no_finding)
    bool_indices = np.logical_and(bool_indices, np.logical_not(uncertain))
    bool_indices = np.logical_and(bool_indices, frontal)
    bool_indices = np.logical_and(bool_indices, fine_sizes)

    indices_list = []

    columns = ["No Finding"] + patho

    rng = np.random.default_rng()
    for col in columns:
        if col != "No Finding":
            p_indices = np.where(np.logical_and((labels[col].values == 2), bool_indices))
            print(col, p_indices[0].shape[0])

            sample_ids = rng.choice(p_indices[0].shape[0], per_class, replace=False)

        else:
            p_indices = np.where(np.logical_and((labels[col].values == 1), bool_indices))
            print(col, p_indices[0].shape[0])

            sample_ids = rng.choice(p_indices[0].shape[0], no_finding_lim, replace=False)

        p_chosen = p_indices[0][sample_ids]


        p_chosen_bool = np.zeros((labels.shape[0],))
        p_chosen_bool[p_indices[0][sample_ids]] = 1

        bool_indices = np.logical_and(bool_indices, np.logical_not(p_chosen_bool))

        indices_list.append(p_chosen)

    indices = np.concatenate(indices_list)

    exclude = [p for p in ds.PATHOLOGIES if p not in patho]

    chosen = labels.drop(exclude, axis=1).iloc[indices]

    for p in patho:

        chosen.loc[chosen[p] == 2, p] = 1

    """
    y, x = np.where(chosen_train[patho] == 2)
    patho_labels = chosen_train[patho].copy()
    patho_labels.iloc[y,x] = 1
    chosen_train[patho] = patho_labels

    chosen_val = labels.drop(exclude, axis=1).iloc[indices_val]
    #y, x = np.where(chosen_val[patho] == 2)
    #patho_labels = chosen_val[patho].copy()
    #patho_labels.iloc[y,x] = 1
    #chosen_val[patho] = patho_labels
    """

    #sample_idx = rng.choice(size, , replace=False)

    #patho_adjust = [''.join(p.split(' ')) for p in patho]

    rng = np.random.default_rng()
    split = rng.choice(chosen.shape[0], int(chosen.shape[0]*(1-val_split)), replace=False)

    chosen_bool = np.zeros((chosen.shape[0],), dtype=bool)
    chosen_bool[split] = True
    chosen_train = chosen.loc[chosen_bool,:]
    chosen_val = chosen.loc[np.logical_not(chosen_bool),:]


    save_path_train = labels_path.split('train.csv')[0] + f'train-{chosen_train.shape[0]}-' + 'reduced'  + '.csv'
    save_path_val = labels_path.split('train.csv')[0] + f'valid-{chosen_val.shape[0]}-' + 'reduced'  + '.csv'


    # patho_adjust = [''.join(p.split(' ')) for p in patho]

    #save_path_train = (
    #    labels_path.split("train.csv")[0]
    #    + f"train-{(len(patho)+1)*train_per_class}-"
    #    + "reduced"
    #    + ".csv"
    #)
    #save_path_val = (
    #    labels_path.split("train.csv")[0]
    #    + f"valid-{(len(patho)+1)*val_per_class}-"
    #    + "reduced"
    #    + ".csv"
    #)

    print(save_path_train)
    print(save_path_val)
    chosen_train.to_csv(save_path_train, index=False)
    chosen_val.to_csv(save_path_val, index=False)


def subsample_dataset(
    labels_path,
    amount=None,
    factor=0.1,
    only_frontal=True,
    val=False,
    val_amount=1000,
    copy=False,
    path_select=None,
):

    labels = pd.read_csv(labels_path)

    if only_frontal:
        # paths = [path for path in labels.Path if path.endswith('frontal.jpg')]
        frontal_bool = labels["Frontal/Lateral"] == "Frontal"
        labels.Path[
            [True if path.endswith("frontal.jpg") else False for path in labels.Path]
        ]
    else:
        paths = labels.Path

    size = len(paths)

    if amount is None:
        amount = int(np.round(size * factor))

    if not val:
        val_amount = 0

    print(f"From {size} selecting {amount} samples and {val_amount} for validation")

    def subsample():
        rng = np.random.default_rng()
        sample_ids = rng.choice(size, amount + val_amount, replace=False)
        sample_ids.sort(0)
        sample_ids = iter(sample_ids)
        next_id = next(sample_ids)
        for i, path in enumerate(paths):
            if i == next_id:
                yield next_id
                try:
                    next_id = next(sample_ids)
                except StopIteration as e:
                    return

    samples = np.array(list(subsample()))

    if only_frontal:
        save_as_train = "-subsampled-frontal-" + str(amount) + ".csv"
        save_as_val = (
            "-subsampled-frontal-" + str(amount) + "-val-" + str(val_amount) + ".csv"
        )
    else:

        save_as_train = "-subsampled-" + str(amount) + ".csv"
        save_as_val = (
            "-subsampled-" + str(amount) + "-valid-" + str(val_amount) + ".csv"
        )

    save_path_train = labels_path.split(".csv")[0] + save_as_train

    print(f"Saving train to {save_path_train}")

    if val:
        save_path_val = labels_path.split(".csv")[0] + save_as_val
        print(f"Saving val to {save_path_val}")
        rng = np.random.default_rng()
        train_ids = rng.choice(len(samples), amount, replace=False)
        train_ids.sort()
        train_bools = np.zeros((amount + val_amount), dtype=np.bool_)
        for i in train_ids:
            train_bools[i] = 1
        print(len(train_ids))

        labels.iloc[samples[train_bools]].to_csv(save_path_train, index=False)
        labels.iloc[samples[np.logical_not(train_bools)]].to_csv(
            save_path_val, index=False
        )
    else:
        labels.iloc[samples].to_csv(save_path_train, index=False)


def copy_files(csv, dir_name=None):

    paths = pd.read_csv(csv).Path

    root_dir = os.path.split(csv)[0]

    for path in paths:
        copy_from = os.path.join(os.path.split(root_dir)[0], path)
        """
        if dir_name is None:
            copy_to = os.path.join(
                root_dir,
                os.path.split(csv)[1].split(".")[0],
                path.split("CheXpert-v1.0/train/")[1],
            )
        else:
            copy_to = os.path.join(
                root_dir, dir_name + path.split("CheXpert-v1.0/train/")
            )
        """
        copy_to = os.path.join('../Data/reduced',  path)
        
        if not os.path.isdir(os.path.split(copy_to)[0]):
            os.makedirs(os.path.split(copy_to)[0])
        
        #print(copy_to)
        copyfile(copy_from, copy_to)


def rename_paths(csv, new_head):
    labels = pd.read_csv(csv)

    for i in range(labels.shape[0]):
        labels.loc[i, "Path"] = os.path.join(
            new_head, labels.iloc[i]["Path"].split("CheXpert-v1.0/train/")[1]
        )

    labels.to_csv(
        os.path.join(csv.split("CheXpert-v1.0/")[0], ".".join((new_head, "csv"))),
        index=False,
    )


def adapt_paths_to_dir(
    csv, stub="CheXpert-v1.0/train", replace_with="CheXpert-v1.0-reduced/train"
):
    labels = pd.read_csv(csv)
    labels["Path"] = labels["Path"].apply(lambda x: replace_with + x.split(stub)[1])
    print(labels["Path"])
    labels.to_csv(csv, index=False)

def change_ending(csv, ending='.png'):
    labels = pd.read_csv(csv)
    labels['Path'] = labels['Path'].apply(lambda x: '.'.join(x.split('.')[:-1]) + '.png')
    labels.to_csv(csv, index=False)
    