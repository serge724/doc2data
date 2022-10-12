import numpy as np
from doctr.models import ocr_predictor

ocr_model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained = True)

def run_doctr_ocr(image):
    "Runs docTR OCR on a single image."
    result = ocr_model([np.array(image)])
    page = result.export()['pages'][0]
    tokens = []
    c = 0
    for b in page['blocks']:
        for l in b['lines']:
            for w in l['words']:
                tokens.append({
                    'id': c,
                    'bbox': w['geometry'][0] + w['geometry'][1],
                    'text': w['value']
                })
                c += 1
    return tokens
