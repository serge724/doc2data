# SPDX-FileCopyrightText: 2022-present Sergej Levich <sergej.levich@gmail.com>
#
# SPDX-License-Identifier: Apache-2.0

"""Reading PDF files and creating collections."""

import os
import pickle
import logging
import numpy as np
import pandas as pd
import fitz
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from PIL import Image, ImageDraw

from doc2data.utils import (
    get_pcnt_chars_corrupted,
    normalize_bbox,
    denormalize_bbox,
    convert_to_rgb,
)

logger = logging.getLogger(__name__)


class Page:

    """Class representing individual pages from a pdf file.

    This class allows accessing the contents of a page using the pymupdf package.
    It also contains additional attributes that describe the page content.
    All attribute values are obtained using the pymupdf interface therefore relying
    on its accuracy.

    Attributes:
        pt_height: Height of the page in points (1 point = 1/72 inch).
        pt_width: Width of the page in points (1 point = 1/72 inch).
        number: Page number within the pdf file.
        content_type: Type of information contained in the page. One of:

            - text: Pure text.
            - images: One or more images.
            - mixed: A combination of text and images.
            - recovered_image: If pymupdf does not detect any content, the page is
                rendered as an image.
        n_tokens: Number of tokens identified by pymupdf. Roughly corresponds to words.
        n_images: Number of images identified by pymupdf.
        pcnt_chars_corrupted: Number of characters that could not be correctly decoded.
        path_to_pdf: Path to pdf file containing the page.
    """

    def __init__(
        self,
        pt_height,
        pt_width,
        number,
        content_type,
        n_tokens,
        n_images,
        pcnt_chars_corrupted,
        path_to_pdf,
    ):
        self.pt_height = pt_height
        self.pt_width = pt_width
        self.number = number
        self.content_type = content_type
        self.n_tokens = n_tokens
        self.n_images = n_images
        self.pcnt_chars_corrupted = pcnt_chars_corrupted
        self.path_to_pdf = path_to_pdf

    def read_contents(self, types, force_rgb=None, dpi=None):

        """Read page contents via the pymupdf interface.

        This opens the pdf file with pymupdf and extracts the requested
        content. Multiple content types can be provided simultaneously.

        Args:
          types:
            String or list of strings indicating which contents should be
            returned. Possible content types are:

              - tokens: Tokens with bounding boxes. A token rougly corresponds
                  to a word.
              - text: String containg all tokens in the reading order which
                  is recovered by pymupdf.
              - images: Images with bounding boxes.
              - page_image: Entire page as one image.
              - raw_dict: Raw output from pymupdf.
          force_rgb:
            Only if 'page_image' in types: Coverts to RGB and adds white backgound.
          dpi:
            Only if 'page_image' in types: Resolution to use when converting pdf to image.

        Returns:
          A dictionary with the requested contents. If a single type was requested,
            it is returned directly.

        Examples:
          >>> from doc2data.pdf import PDFFile
          >>> pdf_file = PDFFile('path_to_pdf')
          >>> pdf_file.parse_pages()
          >>> page = pdf_file.processed_pages[0]
          >>> page.read_contents('page_image')
          >>> page.read_contents(['tokens', 'images'])

        Raises:
          ValueError: If one or more requested types are not recognized.
        """

        allowed_types = set(("tokens", "text", "images", "page_image", "raw_dict"))
        if isinstance(types, str):
            types = [types]
        if set(types).difference(allowed_types) != set():
            raise ValueError("Content type not recognized")

        with fitz.open(self.path_to_pdf) as pdf:
            page = pdf.load_page(self.number)
            contents = {}
            if "tokens" in types:
                records = []
                words = page.get_text(option="words")
                words.sort(key=lambda x: x[5:8])
                for i, word in enumerate(words):
                    records.append(
                        {
                            "id": i,
                            "bbox": normalize_bbox(
                                word[0:4], self.pt_width, self.pt_height
                            ),
                            "text": word[4],
                        }
                    )
                contents["tokens"] = records
            if "text" in types:
                contents["text"] = page.get_text(option="text")
            if "images" in types:
                records = []
                image_references = page.get_images(full=True)
                for i, ref in enumerate(image_references):
                    pix = fitz.Pixmap(pdf.extract_image(ref[0])["image"])
                    if ref[1] > 0:  # add /SMask or /Mask if exists
                        mask = fitz.Pixmap(pdf.extract_image(ref[1])["image"])
                        pix = fitz.Pixmap(pix, mask)
                    mode = "RGBA" if pix.alpha else "RGB"
                    image = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
                    bbox = page.get_image_bbox(ref)
                    normalized_bbox = normalize_bbox(
                        list(bbox), self.pt_width, self.pt_height
                    )
                    records.append({"id": i, "bbox": normalized_bbox, "image": image})
                contents["images"] = records
            if "page_image" in types:
                pix = page.get_pixmap(alpha=True, dpi=dpi)
                mode = "RGBA" if pix.alpha else "RGB"
                image = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
                if image.mode == "RGBA" and force_rgb:
                    image = convert_to_rgb(image, add_white_background=True)
                contents["page_image"] = image
            if "raw_dict" in types:
                contents["raw_dict"] = page.get_text(option="rawdict")

        if len(types) == 1:
            contents = contents.pop(types[0])

        return contents

    def show_page_image(self, dpi=None, show_bboxes=False, bbox_color="red"):
        """Wrapper for Page.read_contents to show page as image.

        Args:
          dpi: Resolution to use when converting pdf to image.
          show_bboxes: Boolean indicating whether to draw bounding boxes aroung
            the tokens and images.
          bbox_color: Color of the bounding boxes.

        Returns:
            A PIL image of the document page.
        """

        contents = self.read_contents(
            ["page_image", "tokens", "images"], force_rgb=True, dpi=dpi
        )
        image = contents["page_image"]
        if show_bboxes:
            elements = contents["tokens"]
            elements.extend(contents["images"])
            draw = ImageDraw.Draw(image)
            for element in elements:
                scaled_bbox = denormalize_bbox(
                    element["bbox"], image.width, image.height
                )
                draw.rectangle(scaled_bbox, outline=bbox_color)

        return image


class PDFFile:
    """Class representing an individual pdf file.

    Attributes:
      path_to_pdf: Path to pdf file.
      file_name: File name.
      n_pages: Number of pages in the pdf file.
      parsed_pages: List of Page objects providing an interface to each page.
      loaded_successfully: Boolan indicating if pdf file could be processed.
    """

    def __init__(self, path_to_pdf):
        self.path_to_pdf = path_to_pdf
        self.file_name = os.path.basename(path_to_pdf)
        self.n_pages = None
        self.parsed_pages = None
        self.loaded_successfully = None

    def __getitem__(self, item):
        return self.parsed_pages[item]

    def __repr__(self):
        return (
            f"<{self.__module__}.{self.__class__.__name__}"
            f" with {self.n_pages} pages at {hex(id(self))}>"
        )

    def open_fitz_document(self):
        """Opens the pdf file as fitz object."""
        assert self.file_name.lower().endswith(
            ".pdf"
        ), f"File {self.file_name} is not a PDF and will be ignored"
        return fitz.open(filename=self.path_to_pdf)

    def parse_pages(self):
        """Iterates over pages and instantiates Page objects.

        Raises:
          AssertionError: If the file does not have a .pdf extension.
          RuntimeError: If pymupdf fails to open the file.
        """

        try:
            with self.open_fitz_document() as pdf:
                self.n_pages = pdf.page_count
                self.parsed_pages = {}
                for i, page in enumerate(pdf):
                    page_content = page.get_text("rawdict")
                    block_types = np.array([b["type"] for b in page_content["blocks"]])
                    if len(block_types) == 0:
                        content_type = "recovered_image"
                    elif all(block_types == 0):
                        content_type = "text"
                    elif all(block_types == 1):
                        content_type = "images"
                    else:
                        content_type = "mixed"

                    parsed_page = Page(
                        pt_height=page_content["height"],
                        pt_width=page_content["width"],
                        number=page.number,
                        content_type=content_type,
                        n_tokens=len(page.get_text_words()),
                        n_images=sum(block_types),
                        pcnt_chars_corrupted=get_pcnt_chars_corrupted(page),
                        path_to_pdf=self.path_to_pdf,
                    )

                    self.parsed_pages[i] = parsed_page
                self.loaded_successfully = True

        except AssertionError as assertion_error:
            self.loaded_successfully = False
            logger.warning(assertion_error)
        except RuntimeError:  # check for errors raised by pymupdf
            self.loaded_successfully = False
            logger.warning(
                "File %s could not be read and will be ignored", self.file_name
            )


class PDFCollection:
    """Creates a collection from pdf files that are stored in a directory.

    This class serves multiple purposes: First, it allows to parse pdf files
    from a directory to create a collection. Second, it provides overview
    information about the collection on page level. Third, it allows reading
    contents from individual pdfs through a dictionary with PDFFile objects.

    The PDFCollection class does not store any files itself. Instead, it only
    keeps track of the files present in the target folder via a dictionary.
    Therefore, once a collection is created, files should not be removed from
    the source folder.

    Examples:
      >>> from doc2data.pdf import PDFCollection
      >>> pdf_collection = PDFCollection('path_to_files') # create collection
      >>> pdf_collection.parse_files() # populate collection
      >>> print(pdf_collection.overview) # inspect collection

    Attributes:
      path_to_files: Path to source directory containing the pdf files.
      pdfs: A dictionary containing a PDFFile object for each file.      
      ignored_files: A list of file names that could not be processed.
      overview: A Pandas DataFrame containing summary information on page level.
    """

    def __init__(self, path_to_files):
        self.path_to_files = path_to_files
        self.pdfs = None
        self.ignored_files = None
        self.overview = None

    def count_source_files(self):
        """Counts files in the source directory."""
        return len(os.listdir(self.path_to_files))

    def parse_single_file(self, file_name):
        """Instantiates PDFFile object and parses pages."""

        pdf = PDFFile(os.path.join(self.path_to_files, file_name))
        pdf.parse_pages()
        return pdf

    def parse_files(self, use_multiprocessing=True):
        """Populates the collection with PDFFile objects based on source directory.

        Additionally, information on the content is extracted and recorded on
        page level.

        Args:
          use_multiprocessing: Use multiprocessing to speed up population using
            all available cores
        """

        logger.info("Parsing files in %s", self.path_to_files)

        all_files = os.listdir(self.path_to_files)

        if use_multiprocessing:
            pdf_list = process_map(self.parse_single_file, all_files, chunksize=10)
        else:
            pdf_list = list(map(self.parse_single_file, tqdm(all_files)))

        self.pdfs = {i.file_name: i for i in pdf_list if i.loaded_successfully}
        self.ignored_files = [
            i.file_name for i in pdf_list if not i.loaded_successfully
        ]

        overview = []
        for pdf in self.pdfs.values():
            if pdf.loaded_successfully:
                for page in pdf.parsed_pages.values():
                    overview.append(
                        {
                            "file_name": pdf.file_name,
                            "page_nr": page.number,
                            "pt_height": page.pt_height,
                            "pt_width": page.pt_width,
                            "size": page.pt_height * page.pt_width,
                            "content_type": page.content_type,
                            "n_tokens": page.n_tokens,
                            "n_images": page.n_images,
                            "pcnt_chars_corrupted": page.pcnt_chars_corrupted,
                        }
                    )
        self.overview = pd.DataFrame(overview)

        logger.info(
            "%s files parsed, %s files ignored", len(all_files), len(self.ignored_files)
        )

    def save(self, file_path, overwrite=False):
        """Serializes collection to a file with pickle.

        Path directories are created if they do not exist.

        Args:
          file_path: File path where to save the collection.
          overwrite: Set to True if an existing collection should be overwritten.
        """
        if os.path.exists(file_path) and not overwrite:
            raise FileExistsError("Collection already exists")
        if os.path.dirname(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(file_path):
        """Loads serialized collection.

        Args:
          file_path: File path from where to load the collection

        Returns:
          A PDFCollection object.
        """

        with open(file_path, "rb") as file:
            dataset = pickle.load(file)

        return dataset
