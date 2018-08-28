import os
import sys
import skimage.io
import numpy as np
import keras
import random

"""
ImageDataset creates a Matterport dataset for a directory of
images in order to ensure compatibility with benchmarking tools
and image resizing for networks.

Directory structure must be as follows:

$base_path/
    test_indices.npy
    train_indices.npy
    depth_ims/ (Depth images here)
        image_000000.png
        image_000001.png
        ...
    color_ims/ (Color images here)
        image_000000.png
        image_000001.png
        ...
    modal_segmasks/ (GT segmasks here, one channel)
        image_000000.png
        image_000001.png
        ...
"""

class ImageDataset(object):

    def __init__(self, base_path, imset):
        assert base_path != "", "You must provide the path to a dataset!"

        self.data_path = os.path.join(base_path, imset)
        self._class_labels = os.listdir(self.data_path)
        self._num_classes = len(self._class_labels)
        self._num_images = len(os.listdir(os.path.join(self.data_path, self._class_labels[0])))
        self.triple_info = []

    def generate_triple_from_ind(self, index):
        class_ind = index % self._num_images**2
        rem_ind = index - class_ind
        image1_ind = int((index - class_ind)/self._num_images)


    def generate_triple(self, pos=False):
        class1 = random.choice(self._class_labels)
        folder1 = os.path.join(self.data_path, class1)
        im1 = os.path.join(folder1, random.choice(os.listdir(folder1)))
        if pos:
            im2 = os.path.join(folder1, random.choice(os.listdir(folder1))) 
            # while im2 != im1:
            #     im2 = os.path.join(folder1, random.choice(os.listdir(folder1)))
            label = 1
        else:
            class2 = random.choice(self._class_labels)
            while class2 == class1:
                class2 = random.choice(self._class_labels)
            folder2 = os.path.join(self.data_path, class2)
            im2 = os.path.join(folder2, random.choice(os.listdir(folder2)))
            label = 0
        return (im1, im2, label)

    def add_triple(self, path1, path2, label):
        self.triple_info.append({
            "im1": path1,
            "im2": path2,
            "label": label
        })

    def prepare(self, size):
        for i in range(size):
            self.add_triple(*self.generate_triple(True))
            self.add_triple(*self.generate_triple(False))

    def load_im1(self, image_id):
        # loads image from path
        image = skimage.io.imread(self.triple_info[image_id]['im1'])

        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image

    def load_im2(self, image_id):
        # loads image from path
        image = skimage.io.imread(self.triple_info[image_id]['im2'])

        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image

    def load_label(self, image_id):
       return self.triple_info[image_id]['label']

    @property
    def triples(self):
        return self.triple_info


class DataGenerator(keras.utils.Sequence):
    def __init__(self, im_dataset, batch_size=32, dim=(32,32,32), shuffle=True):
        """Initialization"""
        self.im_dataset = im_dataset
        self.batch_size = batch_size
        self.dim = dim
        self.shuffle = shuffle
        self.on_epoch_end()

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, Y, z = self.__data_generation(indices)

        return [X, Y], z

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indices = np.arange(len(self.im_dataset.triples))
        if self.shuffle == True:
            np.random.shuffle(self.indices)

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.im_dataset.triples) / self.batch_size))

    def __data_generation(self, indices):
        """Generates data containing batch_size samples,  X : (n_samples, *dim, n_channels) """
        # Initialization
        X = np.empty((self.batch_size, *self.dim), dtype=np.uint8)
        Y = np.empty((self.batch_size, *self.dim), dtype=np.uint8)
        z = np.empty((self.batch_size), dtype=np.uint8)

        # Generate data
        for i, ind in enumerate(indices):
            # Store sample
            X[i,] = self.im_dataset.load_im1(ind)
            Y[i,] = self.im_dataset.load_im2(ind)

            # Store class
            z[i] = self.im_dataset.load_label(ind)

        return X, Y, z
