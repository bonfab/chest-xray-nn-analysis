import pandas as pd
import os
from torch.utils.data import Dataset
from torchvision import transforms, utils
from math import isnan
import numpy as np
import torch
import image_processing as ip
import imagesize
from PIL import Image
import custom_transformations as custom_transforms

PATHOLOGIES = [
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]


def concat_imgs_1channel(paths, length, trans, size=256):

    img_concat = np.empty((length * size * size,), dtype=np.uint8)
    for i, img in enumerate(ip.get_all_img_parallel(paths)):
        img_concat[i * size * size : (i + 1) * size * size] = (
            trans(img).flatten().numpy()
        )
    return img_concat


def determine_normalization(paths, length, trans, size, channel=1):
    imgs_array = concat_imgs_1channel(paths, length, trans, size)
    return np.mean(imgs_array), np.std(imgs_array, ddof=1)


def get_labels(pd_data):

    for column in pd_data.columns:

        if column in PATHOLOGIES:
            pd_data[column] = [
                category + 1 if not isnan(category) else 0
                for category in pd_data[column]
            ]
        if column == "No Finding":
            pd_data[column] = [
                category if not isnan(category) else 0 for category in pd_data[column]
            ]
    # print(pd_data)
    return torch.tensor(pd_data[PATHOLOGIES + ["No Finding"]].values)


def extract_paths(root_dir, csv_paths):
    return (os.path.join(os.path.split(root_dir)[0], path) for path in csv_paths)


def path_to_img(root_dir, csv_path):
    return os.path.join(os.path.split(root_dir)[0], csv_path)


def determine_sizes(chex_root_dir, csvs):

    csv_paths = np.concatenate([pd.read_csv(csv).Path.values for csv in csvs])

    sizes = np.empty((csv_paths.shape[0], 2))

    for i, path in enumerate(extract_paths(chex_root_dir, csv_paths)):

        sizes[i] = np.array(imagesize.get(path))

    mins = sizes.min(axis=0)
    min_x = mins[0]
    min_y = mins[1]

    maxs = sizes.min(axis=0)
    max_x = maxs[0]
    max_y = maxs[1]

    return min_y, max_y, min_x, max_x, sizes


RESIZE_CROP_GRAY = transforms.Compose(
    [
        transforms.Resize(342, Image.LANCZOS),
        transforms.CenterCrop(256),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ]
)


RESIZE_CROP_GRAY = transforms.Compose([transforms.Resize(256, Image.LANCZOS), transforms.CenterCrop(256), transforms.Grayscale(), transforms.ToTensor(), transforms.Normalize(mean=[0.5584914588061902], std=[0.2683745711250149])])

GAMMA = transforms.Compose([custom_transforms.GammaTransform(gamma=0.5), transforms.Resize(256, Image.LANCZOS), transforms.CenterCrop(256), transforms.Grayscale(), transforms.ToTensor()])

class BinaryClassification(Dataset):
    def __init__(self, csv, trans=RESIZE_CROP_GRAY, output_size=(256, 1)):

        """
        Prepares data for binary classification (pathology present/abesnt).

        Args:
            csv: path to csv file to be used for data loading.

            trans: Transformation that outputs a tensor.

            output_size: output tensor dimensions. First value: height/width of tensor. Second: number of channels.
        """

        data = pd.read_csv(csv)
        self.output_size = output_size
        self.transform = trans

        # arrange paths for image loading

        root_dir = os.path.split(csv)[0]
        self.paths = data["Path"].apply(lambda x: path_to_img(root_dir, x)).values

        # get according pathologies and their labels

        patho = [cl for cl in data.columns if cl in PATHOLOGIES]
        self.labels = torch.tensor(data[patho + ["No Finding"]].values).float()
        self.num_pathologies = len(patho)

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        
        img = Image.open(self.paths[index])

        return self.transform(img), self.labels[index]


class ResizeCropGray(Dataset):
    def __init__(self, chex_root_dir, csv, output_size=256):

        data = pd.read_csv(csv)
        self.paths = np.empty((data.shape[0],), dtype=object)
        for i, path in enumerate(extract_paths(chex_root_dir, data.Path)):
            self.paths[i] = path

        patho = [cl for cl in data.columns if cl in PATHOLOGIES]

        self.labels = torch.tensor(data[patho + ["No Finding"]].values).float()
        self.num_pathologies = len(patho)

        self.transform = transforms.Compose(
            [
                transforms.Resize(output_size, Image.LANCZOS),
                transforms.CenterCrop(output_size),
                transforms.Grayscale(),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):

        img = Image.open(self.paths[index])

        return self.transform(img), self.labels[index]


class NaiveResizeGray(Dataset):
    def __init__(
        self, chex_root_dir, csv, center_crop_size, output_size, mean=None, std=None
    ):

        data = pd.read_csv(csv)
        self.paths = np.empty((data.shape[0],), dtype=object)
        for i, path in enumerate(extract_paths(chex_root_dir, data.Path)):
            self.paths[i] = path

        patho = [cl for cl in data.columns if cl in PATHOLOGIES]

        self.labels = torch.tensor(data[patho + ["No Finding"]].values).long()
        self.num_pathologies = len(patho)

        if mean is not None:
            self.mean = mean
            self.std = std
        else:
            self.mean, self.std = determine_normalization(
                self.paths,
                data.shape[0],
                transforms.Compose(
                    [
                        custom_transforms.NaiveGrayTransform(
                            center_crop_size, output_size
                        ),
                        transforms.ToTensor(),
                    ]
                ),
                output_size,
            )
        # print(self.mean, self.std)

        self.transform = transforms.Compose(
            [
                custom_transforms.NaiveGrayTransform(center_crop_size, output_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[self.mean], std=[self.std]),
            ]
        )

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):

        img = ip.load_img_rgb(self.paths[index])

        return self.transform(img), self.labels[index].long()
