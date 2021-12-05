from flask import Flask, jsonify, request
from flask_cors import CORS
# from flask_ngrok import run_with_ngrok
import time
import cv2
import numpy as np

from services_utils import *

device = select_device('0')
# device = select_device('cpu')
# load models
(
    (craft_detect, model_recognition),
    (mmocr_recog, pan_detect, classifyModel_level1, mapping_checkpoints),
    (chinh_model, model_step, labels_end, labels_branch, dict_middle, dict_step),
    model_object_detect,
) = load_model(
    device=device,
)

keywords = load_keywords(keywords_fn)
spell = None#load_spell_checker(correct_corpus_fn)

infer = Infer(
    craft_detect, model_recognition,
    mmocr_recog, pan_detect, classifyModel_level1, mapping_checkpoints,
    chinh_model, model_step, labels_end, labels_branch, dict_middle, dict_step,
    model_object_detect,
    keywords, spell,
    device,
)


def get_app():
    app = Flask(__name__)
    CORS(app)
    # run_with_ngrok(app)

    @app.route('/api/milk_pred', methods=['POST'])
    def milk_pred():
        image_files = request.files.getlist('images')

        t1 = time.time()
        images = []
        for image_file in image_files:
            img = cv2.imdecode(
                np.frombuffer(image_file.read(), np.uint8), 3,
            )
            # img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
            images.append(img)

        try:
            results = infer.run(images=images)
            res = {
                'message': 'OK',
                'success': True,
                'data': results,
                'time': f"{time.time() - t1}s"
            }
        except Exception as e:
            # raise e
            res = {
                'message': f'Error {str(e)}',
                'success': False,
            }
        return jsonify(res)

    return app


if __name__=='__main__':
    app = get_app()
    app.run('0.0.0.0', port=5000, )
    # app.run()
