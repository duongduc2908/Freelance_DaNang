pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.7.0/index.html

cd classifyText/newpan
bash compile.sh
cd ../../

if ! [ -f "checkpoints.zip" ]
then
    gdown --id 1SjL0xj2rQRSEL37EIDtkvgVc4xfMTgO1
    unzip checkpoints.zip
    rm -f checkpoints.zip
fi
pip install pyspellchecker==0.6.2
pip install ngram
pip install scikit-image
pip install lmdb
pip install -U scikit-learn
pip install tensorflow==2.4.1
pip install pandas
pip install tqdm
pip install seaborn
pip install natsort
pip install opencv-python==4.1.2.30
pip install flask_cors
pip install flask