cd classifyText/newpan
bash compile.sh
cd ../../

if ! [ -f "checkpoints.zip" ]
then
    # gdown --id 1269bExpsdHUI8g3wROapdOxboaKoyPNr
    gdown --id 1269bExpsdHUI8g3wROapdOxboaKoyPNr
    unzip checkpoints.zip
    rm -f checkpoints.zip
fi
