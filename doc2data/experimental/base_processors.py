import os
import numpy as np
from copy import copy
from doc2data.utils import load_image

try:
    import tensorflow as tf
    _tf_available = True
except ImportError:
    _tf_available = False

try:
    import torch
    _torch_available = True
except ImportError:
    _torch_available = False

if _tf_available:
    import tensorflow as tf
elif _torch_available:
    import torch
else:
    raise ImportError(
        'Neither TensorFlow nor PyTorch could be loaded. '
        'Please make sure one of the two is installed.'
    )

class DataProcessorMixin:
    """MixIn class for data processing in the training loop."""

    def __init__(
        self,
        instances_df,
        processor = None
    ):
        self.instances_df = instances_df.copy()
        self.processor = processor
        self.inference_mode = None

    def load_instance(self, source):
        pass

    def get_processed_instance(self, source):
        features, labels = self.load_instance(source)
        if self.processor:
            features, labels = self.processor(features, labels, self.inference_mode)

        return features, labels

if _tf_available:

    class DataProcessor(DataProcessorMixin, tf.keras.utils.Sequence):
        """Base data processing pipeline for TensorFlow backend."""

        batch_size = 1

        def __len__(self):
            return int(np.ceil(len(self.instances_df) / self.batch_size))

        def __getitem__(self, idx):
            batch_df = self.instances_df.iloc[idx * self.batch_size:(idx + 1) * self.batch_size]

            batch_features = []
            batch_labels = []
            for i, row in batch_df.iterrows():
                features, labels = self.get_processed_instance(row)
                batch_features.append(features)
                batch_labels.append(labels)
            if isinstance(features, list):
                batch_features = [
                    np.stack([i[0] for i in batch_features], axis = 0),
                    np.stack([i[1] for i in batch_features], axis = 0)
                ]
            else:
                batch_features = np.stack(batch_features, axis = 0)

            batch_labels = np.stack(batch_labels, axis = 0)

            return batch_features, batch_labels

        def on_epoch_end(self):
            pass

        def initialize_split(self, data_split, batch_size, shuffle = None, inference_mode = 'auto'):
            """Returns a Keras Sequence object."""

            sequence = copy(self)
            sequence.instances_df = sequence.instances_df.query('data_split == %s'%str([data_split]))
            sequence.batch_size = batch_size
            if shuffle: print('Shuffle not applicable for tf.keras.Sequence object.')
            if inference_mode == 'auto':
                sequence.inference_mode = False if data_split == 'train_set' else True
            else:
                sequence.inference_mode = inference_mode
            return sequence

if _torch_available and not _tf_available:

    class DataProcessor(DataProcessorMixin, torch.utils.data.Dataset):
        """Base data processing pipeline for PyTorch backend."""

        def __len__(self):
            return len(self.instances_df)

        def __getitem__(self, idx):
            # Note: https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
            row = self.instances_df.iloc[idx]
            features, labels = self.get_processed_instance(row)

            return features, labels

        def initialize_split(self, data_split, batch_size, shuffle, inference_mode = 'auto'):
            """Returns a PyTorch DataLoader object."""

            dataset = copy(self)
            dataset.instances_df = dataset.instances_df.query('data_split == %s'%str([data_split]))
            if inference_mode == 'auto':
                dataset.inference_mode = False if data_split == 'train_set' else True
            else:
                dataset.inference_mode = inference_mode
            data_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size = batch_size,
                shuffle = shuffle
            )

            return data_loader
