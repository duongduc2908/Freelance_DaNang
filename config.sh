pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.7.0/index.html #==1.3.12
pip install mmdet==2.16.0 #2.15.1

cd classifyText/mmocr1
pip install -r requirements.txt
pip install -v -e .  # or "python setup.py develop"
export PYTHONPATH=$(pwd):$PYTHONPATH
# rm -rf demo docker docs docs_zh_CN mmocr.egg-info resources tests tools 
# rm MANIFEST.in README.md README_zh-CN.md model-index.yml setup.cfg setup.py

cd classifyText/pan/post_processing/
rm -rf pse.so
make
pip install polygon3
pip install pyclipper
pip install colorlog
python -m pip install Pillow==6.2
cd ../../
# rm -rf base config data_loader imgs result test_img trainer utils
# rm PAN.ipynb README.MD train.py eval.py config.json

if ! [ -f "milk_classification/checkpoints.zip" ]
then
    gdown --id 1-U38UAigrcEgzKEZZgjtF-kHnbZfEYbv
    unzip checkpoints.zip
    rm -f checkpoints.zip
fi
pip install pyspellchecker==0.6.2
pip install ngram==3.3.2
pip install -U scikit-learn
pip install tensorflow-gpu==2.4.1
pip install pandas
pip install tqdm
pip install seaborn
pip install natsort
pip install opencv-python==4.1.2.30
pip install flask_cors
pip install flask

