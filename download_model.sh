cd classifyText/newpan
bash compile.sh
cd ../../

if ! [ -f "checkpoints.zip" ]
then
    gdown --id 1SjL0xj2rQRSEL37EIDtkvgVc4xfMTgO1
    unzip checkpoints.zip
    rm -f checkpoints.zip
fi