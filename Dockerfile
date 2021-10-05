FROM okwrtdsh/anaconda3:10.0-cudnn7

RUN apt-get update -qq \
 && apt-get install --no-install-recommends -y \
	graphviz \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

COPY environment.yml /tmp/environment.yml

WORKDIR /tmp

RUN conda env create -f environment.yml

COPY project /tmp/project

RUN apt update \
 && apt install libgl1-mesa-glx \
 && source activate chinh_test \
	&& cd /tmp/project

CMD cd /tmp