cd classifyText/newpan
bash compile.sh
cd ../../

if ! [ -f "checkpoints.zip" ]
then
    gdown --id 1Org3YirFS74lGIH2T-46YflJYUP8K9l9
    unzip checkpoints.zip
    rm -f checkpoints.zip
fi
