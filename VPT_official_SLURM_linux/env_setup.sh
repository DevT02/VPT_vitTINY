
# conda create -n prompt python=3.7.1 # ORIGINAL
# conda activate prompt
# conda install python=3.8
# if dll error:
conda uninstall pillow
pip install -q tensorflow
# specifying tfds versions is important to reproduce our results
pip install tfds-nightly==4.4.0.dev202201080107
pip install opencv-python
pip install tensorflow-addons
pip install mock

conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.0 -c pytorch

# conda install pytorch==1.8 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch  # ORIGINAL
# conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# python -m pip install detectron2 # ORIGINAL
python -m pip install git+https://github.com/facebookresearch/detectron2.git # NOW WE HAVE DETECTRON
# https://dl.fbaipublicfiles.com/detectron2/wheels/cu110/torch1.7/index.html
pip install opencv-python

conda install tqdm pandas matplotlib seaborn scikit-learn scipy simplejson termcolor
conda install -c iopath iopath


# for transformers
pip install timm==0.4.12
pip install ml-collections

# Optional: for slurm jobs
pip install submitit -U
pip install slurm_gpustat
# downgrade pillow
pip install pillow==9.0.0