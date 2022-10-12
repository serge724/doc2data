import numpy as np
from doc2data.utils import load_image
from doc2data.experimental.base_processors import DataProcessor
from doc2data.experimental.utils import to_categorical

class PageRotationClsProcessor(DataProcessor):
    """Processor for page rotation."""

    def __init__(
        self,
        instances_df,
        processor = None,
        image_size = (224, 224)
    ):
        super().__init__(instances_df, processor)

        self.instances_df['label_string'] = self.instances_df.labels
        self.instances_df.labels, label_index = self.instances_df.labels.factorize(sort = True)
        self.label_dict = {k: v for k, v in enumerate(label_index)}
        self.n_classes = len(self.label_dict)
        self.image_size = image_size

    def load_instance(self, source):
        features = load_image(source.image_path, target_size = self.image_size)
        labels = source.labels
        return features, labels

    def get_processed_instance(self, source):
        features, labels = super().get_processed_instance(source)
        labels = to_categorical(labels, num_classes = self.n_classes)
        return features, labels


class PageCroppingRegProcessor(DataProcessor):
    """Processor for page cropping."""

    def __init__(
        self,
        instances_df,
        processor = None,
        image_size = (224, 224)
    ):
        super().__init__(instances_df, processor)
        self.image_size = image_size

    def load_instance(self, source):
        features = load_image(source.image_path, target_size = self.image_size)
        labels = np.array(source.labels)
        return features, labels

class BBoxClsProcessor(DataProcessor):
    """Processor for bounding box classification."""

    def __init__(
        self,
        instances_df,
        processor = None,
        image_size = (224, 224)
    ):
        super().__init__(instances_df, processor)

        self.instances_df['label_string'] = self.instances_df.labels
        self.instances_df.labels, label_index = self.instances_df.labels.factorize(sort = True)
        self.label_dict = {k: v for k, v in enumerate(label_index)}
        self.n_classes = len(self.label_dict)
        self.image_size = image_size

    def load_instance(self, source):
        features = np.array(source.bbox)
        labels = to_categorical(source.labels, num_classes = self.n_classes)
        return features, labels

class DocumentClsProcessor(DataProcessor):
    """Processor for document classification."""

    def __init__(
        self,
        instances_df,
        modalities,
        processor = None,
        image_size = (224, 224)
    ):
        super().__init__(instances_df, processor)
        self.instances_df['label_string'] = self.instances_df.labels
        self.instances_df.labels, label_index = self.instances_df.labels.factorize(sort = True)
        self.label_dict = {k: v for k, v in enumerate(label_index)}
        self.n_classes = len(self.label_dict)
        self.max_n_pages = np.max([len(i) for i in instances_df.image_paths])
        self.modalities = modalities
        self.image_size = image_size

    def load_instance(self, source):
        instance_features = []
        if 'image' in self.modalities:
            if isinstance(source.image_paths, list):
                features = []
                for path in source.image_paths:
                    page_image = load_image(path, target_size = self.image_size, to_array = True)
                    features.append(page_image)
                for i in range(self.max_n_pages - len(features)):
                    features.append(np.zeros((self.image_size + (3,))))
                features = np.concatenate(features, axis = 0)
            else:
                features = load_image(source.image_paths, self.image_size)
            instance_features.append(features)
        if 'embeddings' in self.modalities:
            if isinstance(source.embeddings_paths, list):
                features = []
                for path in source.embeddings_paths:
                    page_embeddings = np.load(path)
                    if len(page_embeddings) == 0:
                        # Zero-length embeddings?
                        page_embeddings = np.zeros((1, 300))
                    if len(page_embeddings) > 500:
                        page_embeddings = page_embeddings[0:500,]
                    page_embeddings = np.pad(page_embeddings, ((0, 500 - page_embeddings.shape[0]), (0, 0)))
                    features.append(page_embeddings)
                for i in range(self.max_n_pages - len(features)):
                    features.append(np.zeros((500, 300)))
            else:
                features = np.load(path)
            features = np.concatenate(features, axis = 0)
            instance_features.append(features)

        if len(instance_features) == 1:
            instance_features = instance_features[0]
        label = to_categorical(np.array(source.labels), num_classes = self.n_classes)
        return instance_features, label


class TokenClsProcessorLayoutXLM(DataProcessor):
    """Processor for token classification (LayoutXLM fine-tuning)."""

    def __init__(
        self,
        instances_df,
        processor,
        pdf_collection
    ):
        super().__init__(instances_df, processor)
        self.pdf_collection = pdf_collection
        unique_labels = np.unique([i for j in instances_df.labels for i in j])
        self.id2label = {v: k for v, k in enumerate(unique_labels)}
        self.label2id = {k: v for v, k in enumerate(unique_labels)}

    @staticmethod
    def normalize_bbox_layoutxlm(bbox, width, height):
         return [
             int(1000 * (bbox[0] / width)),
             int(1000 * (bbox[1] / height)),
             int(1000 * (bbox[2] / width)),
             int(1000 * (bbox[3] / height)),
         ]

    def load_instance(self, source):
        pdf = self.pdf_collection.pdfs[source.file_name]
        image = pdf.parsed_pages[0].read_contents('page_image', force_rgb = True)
        tokens = source.tokens
        bboxes = source.bboxes
        bboxes = [self.normalize_bbox_layoutxlm(i, image.width, image.height) for i in bboxes]
        labels = [self.label2id[i] for i in source.labels]

        return (image, tokens, bboxes), labels

    def get_processed_instance(self, source):
        features, labels = self.load_instance(source)
        if self.processor:
            instance = self.processor(features, labels, self.inference_mode)
        else:
            instance = (features, labels)

        return instance

    def __getitem__(self, idx):
        row = self.instances_df.iloc[idx]
        instance = self.get_processed_instance(row)

        return instance
