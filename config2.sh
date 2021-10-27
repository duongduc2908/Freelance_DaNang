pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.7.0/index.html

cd classifyText
git clone https://github.com/PhiDCH/new_pan newpan
cd newpan
bash compile.sh
gdown --id 1dXMKs1VAltre6RCtDHYft-2q8oCtC_Yh -O checkpoint/pan.pth.tar
cd ../..

# download recog model
gdown --id 1LVyPRbLm_x4qtFHg3CjEfE18763UQCXl -O textSpotting/textRecognitionCRNN_finetune_v1.pth

if ! [ -f "milk_classification/checkpoints.zip" ]
then
    gdown --id 1-U38UAigrcEgzKEZZgjtF-kHnbZfEYbv
    unzip checkpoints.zip
    rm -f checkpoints.zip
fi
pip install pyspellchecker==0.6.2
pip install ngram==3.3.2
pip install -U scikit-learn
pip install tensorflow==2.4.1
pip install pandas
pip install tqdm
pip install seaborn
pip install natsort
pip install opencv-python==4.1.2.30
pip install flask_cors
pip install flask