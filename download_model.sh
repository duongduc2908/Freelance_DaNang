cd classifyText/newpan
bash compile.sh
cd ../../

if ! [ -f "checkpoints.zip" ]
then
    gdown --id 123ZMlCLpq0CH2HQSMYfXm9Fqoq7b3ABM
    unzip checkpoints.zip
    rm -f checkpoints.zip
fi
