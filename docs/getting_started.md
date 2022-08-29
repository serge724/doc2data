# Getting started

## Creating a collection
A workflow with doc2data begins with creating a collection of PDF files. To do this, place all PDF files into one source directory and instantiate the PDFCollection object.

```python
from doc2data.pdf import PDFCollection
collection = PDFCollection("path_to_files")
```

The collection is now linked to the defined path. No files should be removed from the directory. Count the files to see how many files are available.

```python
collection.count_files()
```

Now we need to parse the files to check if they are readable and to populate the collection. You can disable multiprocessing by setting `use_multiprocessing` to False.

```python
collection.parse_files(use_multiprocessing=True)
```

Parsing the files populates two attributes. `PDFCollection.pdfs` is a dictionary containing a PDFFile object for each successfully parsed file. The keys are the file names. `PDFCollection.overview` is a pandas DataFrame provinding information on page level.

```python
collection.pdfs # dictionary containing PDFFile objects (keys are filenames)
collection.overview # pd.DataFrame containing information on individual pages
```

The collection is now ready to be used for modeling and can be saved and loaded.

```python
collection.save('file_path', overwrite=False) # will create path directories if necessary
collection = PDFCollection.load('file_path')
```

The PDFCollection object does not store any files itself. Instead, it provides a reference dictionary that can be used to read the files when specific contents are requested.

## Reading file contents
The references to the PDF files are stored in the dictionary `PDFCollection.pdfs` so that files can be accessed by their file names. Each file, in turn, has its own dictionary `PDFFile.parsed_pages` containing references to single pages. Each page is represented through a Page object.

```python
pdf_file = collection.pdfs['file_name.pdf']
pdf_file.parsed_pages # dictionary containing Page objects (keys are page numbers)
page = pdf_file[2] # is equivalent to pdf_file.parsed_pages[2]
```

The contents of each page can be extracted through the function `Page.read_contents` (see API reference for possible content types):

```python
content_dict = page.read_contents(types=['tokens', 'images'])
page_image = page.read_contents(types='page_image', dpi=100, force_rgb=True)
```

If only one content type is requested, it is returned directly, otherwise a dictionary is returned. The `dpi` parameter sets the resolution to be used when converting the entire page to an image. Additionally, `force_rgb` can be used to ensure the RGB mode and a white background of the returned PIL image. The image of the page including bounding boxes for text tokens and contained images can be also extracted as follows:

```python
page.show_page_image(dpi=100, show_bboxes=True)
```

