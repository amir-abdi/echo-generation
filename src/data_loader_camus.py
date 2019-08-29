from glob import glob
import numpy as np
import os
import skimage.io as io
from PIL import Image
from prefetch_generator import background
from keras.preprocessing.image import ImageDataGenerator
import random

NUM_PREFETCH = 10
RANDOM_SEED = 123


class DataLoaderCamus:
    def __init__(self, dataset_path, input_name, target_name, condition_name,
                 img_res, target_rescale, input_rescale, condition_rescale, train_ratio, valid_ratio,
                 labels, augment):
        self.dataset_path = dataset_path
        self.img_res = tuple(img_res)
        self.target_rescale = target_rescale
        self.input_rescale = input_rescale
        self.condition_rescale = condition_rescale
        self.input_name = input_name
        self.target_name = target_name
        self.condition_name = condition_name
        self.augment = augment

        patients = sorted(glob(os.path.join(self.dataset_path, 'training', '*')))
        random.Random(RANDOM_SEED).shuffle(patients)
        num = len(patients)
        num_train = int(num * train_ratio)
        num_valid = int(num_train * valid_ratio)

        self.valid_patients = patients[:num_valid]
        self.train_patients = patients[num_valid:num_train]
        self.test_patients = patients[num_train:]
        if train_ratio == 1.0:
            self.test_patients = glob(os.path.join(self.dataset_path, 'testing', '*'))
        print('#train:', len(self.train_patients))
        print('#valid:', len(self.valid_patients))
        print('#test:', len(self.test_patients))
        print('validation set:', self.valid_patients)
        print('test set:', self.test_patients)

        all_labels = {0, 1, 2, 3}
        self.not_labels = all_labels - set(labels)

        data_gen_args = dict(rotation_range=augment['AUG_ROTATION_RANGE_DEGREES'],
                             width_shift_range=augment['AUG_WIDTH_SHIFT_RANGE_RATIO'],
                             height_shift_range=augment['AUG_HEIGHT_SHIFT_RANGE_RATIO'],
                             shear_range=augment['AUG_SHEAR_RANGE_ANGLE'],
                             zoom_range=augment['AUG_ZOOM_RANGE_RATIO'],
                             fill_mode='constant',
                             cval=0.,
                             data_format='channels_last')
        self.datagen = ImageDataGenerator(**data_gen_args)

    def read_mhd(self, img_path, is_gt):
        if not os.path.exists(img_path):
            return np.zeros(self.img_res + (1,))
        img = io.imread(img_path, plugin='simpleitk').squeeze()
        img = np.array(Image.fromarray(img).resize(self.img_res))
        img = np.expand_dims(img, axis=2)

        if is_gt:
            for not_l in self.not_labels:
                img[img == not_l] = 0
        return img

    def _get_paths(self, stage):
        if stage == 'train':
            return self.train_patients
        elif stage == 'valid':
            return self.valid_patients
        elif stage == 'test':
            return self.test_patients

    @background(max_prefetch=NUM_PREFETCH)
    def get_random_batch(self, batch_size=1, stage='train'):
        paths = self._get_paths(stage)

        num = len(paths)
        num_batches = num // batch_size

        for i in range(num_batches):
            batch_paths = np.random.choice(paths, size=batch_size)
            target_imgs, condition_imgs, input_imgs, weight_imgs = self._get_batch(batch_paths, stage)
            target_imgs = target_imgs * self.target_rescale
            input_imgs = input_imgs * self.input_rescale
            condition_imgs = condition_imgs * self.condition_rescale

            yield target_imgs, condition_imgs, input_imgs, weight_imgs

    def get_iterative_batch(self, batch_size=1, stage='test'):
        paths = self._get_paths(stage)

        num = len(paths)
        num_batches = num // batch_size

        start_idx = 0
        for i in range(num_batches):
            batch_paths = paths[start_idx:start_idx + batch_size]
            target_imgs, condition_imgs, input_imgs, weight_imgs = self._get_batch(batch_paths, stage)
            target_imgs = target_imgs * self.target_rescale
            input_imgs = input_imgs * self.input_rescale
            condition_imgs = condition_imgs * self.condition_rescale
            start_idx += batch_size

            yield target_imgs, condition_imgs, input_imgs, weight_imgs

    def _get_batch(self, paths_batch, stage):
        target_imgs = []
        input_imgs = []
        condition_imgs = []
        weight_maps = []

        for path in paths_batch:
            transform = self.datagen.get_random_transform(img_shape=self.img_res)
            head, patient_id = os.path.split(path)
            target_path = os.path.join(path, '{}_{}.mhd'.format(patient_id, self.target_name))
            condition_path = os.path.join(path, '{}_{}.mhd'.format(patient_id, self.condition_name))
            input_path = os.path.join(path, '{}_{}.mhd'.format(patient_id, self.input_name))

            input_img = self.read_mhd(input_path, '_gt' in self.input_name)
            if self.augment['AUG_INPUT']:
                input_img = self.datagen.apply_transform(input_img, transform)
            input_imgs.append(input_img)

            target_img = self.read_mhd(target_path, '_gt' in self.target_name)
            condition_img = self.read_mhd(condition_path, 1)

            if self.augment['AUG_TARGET']:
                if not self.augment['AUG_SAME_FOR_BOTH']:
                    transform = self.datagen.get_random_transform(img_shape=self.img_res)
                target_img = self.datagen.apply_transform(target_img, transform)
                condition_img = self.datagen.apply_transform(condition_img, transform)
            target_imgs.append(target_img)
            condition_imgs.append(condition_img)

            weight_map_condition = self.get_weight_map(condition_img)
            weight_maps.append(weight_map_condition)

        return np.array(target_imgs), np.array(condition_imgs), np.array(input_imgs), np.array(weight_maps)

    def get_weight_map(self, mask):
        # let the y axis have higher variance
        gauss_var = [[self.img_res[0] * 60, 0], [0, self.img_res[1] * 30]]
        x, y = mask[:, :, 0].nonzero()
        center = [x.mean(), y.mean()]

        from scipy.stats import multivariate_normal
        # TODO: not tested for images with different width and height
        gauss = multivariate_normal.pdf(np.mgrid[
                                        0:self.img_res[1],
                                        0:self.img_res[0]].reshape(2, -1).transpose(),
                                        mean=center,
                                        cov=gauss_var)
        gauss /= gauss.max()
        gauss = gauss.reshape((self.img_res[1], self.img_res[0], 1))
        # print(gauss.max())

        # set the gauss value of the main target part to 1
        gauss[mask > 0] = 1

        return gauss
