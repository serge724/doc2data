import os
import json
import numpy as np
import pandas as pd
from functools import partial
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map


class Extractor():
    """Class that implements feature extraction from a PDFCollection object."""

    @staticmethod
    def extract_page_image(pdf, target_dir, format, dpi):
        records = []
        for page in pdf.parsed_pages.values():

            image = page.read_contents('page_image', dpi = dpi, force_rgb = True)

            file_name = '%s_page_%s.%s'%(os.path.splitext(pdf.file_name)[0], page.number, format)
            file_path = os.path.join(target_dir, file_name)
            image.save(file_path, quality = 100)

            records.append({
                'file_name': pdf.file_name,
                'page_nr': page.number,
                'image_path': file_path
            })

        return pd.DataFrame(records)

    def extract_page_images_from_pdfs(
        self,
        pdf_collection,
        target_dir,
        format = 'jpg',
        dpi = None,
        overwrite = False
    ):
        os.makedirs(target_dir, exist_ok = overwrite)
        paths = process_map(
            partial(self.extract_page_image, target_dir = target_dir, format = format, dpi = dpi),
            pdf_collection.pdfs.values(),
            chunksize = 1
        )
        pdf_collection.image_path_df = pd.concat(paths, ignore_index = True)

    @staticmethod
    def extract_tokens(pdf, target_dir, ocr_engine, dpi):
        records = []
        for page in pdf.parsed_pages.values():

            if ocr_engine is not None and (page.n_tokens < 5 or page.pcnt_chars_corrupted > 0.05):

                image = page.read_contents('page_image', dpi = dpi, force_rgb = True)
                tokens = ocr_engine(image)
                source = 'ocr'
            else:
                tokens = page.read_contents('tokens')
                source = 'pdf'

            file_name = '%s_page_%s.json'%(os.path.splitext(pdf.file_name)[0], page.number)
            file_path = os.path.join(target_dir, file_name)
            with open(file_path, 'wt') as file:
                json.dump(tokens, file)

            records.append({
                'file_name': pdf.file_name,
                'page_nr': page.number,
                'tokens_path': file_path,
                'tokens_source': source
            })

        return pd.DataFrame(records)

    def extract_tokens_from_pdfs(self, pdf_collection, target_dir, ocr_engine = None, dpi = None, overwrite = False):
        os.makedirs(target_dir, exist_ok = overwrite)
        paths = map(
            partial(self.extract_tokens, target_dir = target_dir, ocr_engine = ocr_engine, dpi = dpi),
            pdf_collection.pdfs.values()
        )
        paths = [i for i in tqdm(paths, total = len(pdf_collection.pdfs.values()))]
        pdf_collection.tokens_path_df = pd.concat(paths, ignore_index = True)

    def generate_spacy_embeddings(self, pdf_collection, target_dir, spacy_nlp, overwrite = False):
        os.makedirs(target_dir, exist_ok = overwrite)

        records = []
        for i, row in tqdm(pdf_collection.tokens_path_df.iterrows(), total = len(pdf_collection.tokens_path_df)):

            # load
            with open(row.tokens_path, 'rt') as file:
                tokens = json.load(file)

            # process
            token_sequence = [i['text'] for i in tokens]
            token_sequence = ' '.join(token_sequence)
            nlp_content = spacy_nlp(token_sequence)

            # write
            file_name = '%s_page_%s.npy'%(os.path.splitext(row.file_name)[0], row.page_nr)
            file_path = os.path.join(target_dir, file_name)
            embedding_sequence = np.array([token.vector for token in nlp_content])
            embedding_sequence.shape
            np.save(file_path, embedding_sequence)

            # record
            records.append({
                'file_name': row.file_name,
                'page_nr': row.page_nr,
                'embeddings_path': file_path,
                'n_words': len(tokens),
                'n_spacy_tokens': len(nlp_content),
                'n_valid_embeddings': sum([i.has_vector for i in nlp_content])
            })

        records = pd.DataFrame(records)
        records['pcnt_identified'] = np.round(records.n_valid_embeddings / records.n_spacy_tokens, 2)
        pdf_collection.embeddings_path_df = records
