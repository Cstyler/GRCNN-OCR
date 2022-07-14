# for vast.ai
sudo apt install htop git libgl1-mesa-dev zip -y
ssh-keygen
cat /root/.ssh/id_rsa.pub
# Go here and add key: https://bitbucket.org/account/settings/ssh-keys/
git clone git@bitbucket.org:anti-fraud/separator.git
cd separator
yes|pip install -r requirements.txt 
conda install -c conda-forge accimage

git config --global user.email "hanchik97@gmail.com"
git config --global user.name "Khan"
# Install Python extension in VSCode

##

sudo apt-get install python3-pip -y
yes|sudo pip3 install virtualenv

virtualenv inpenv --python=/usr/bin/python3
source venvs/inpenv/bin/activate
yes|pip install -r requirements.txt 

##
torchrun --nproc_per_node=1 train.py \
--lr-scheduler cosineannealinglr --lr-warmup-method linear \
--auto-augment ta_wide --random-erase 0.1 --weight-decay 0.00002 \
--norm-weight-decay 0.0 --label-smoothing 0.1 --mixup-alpha 0.2 --cutmix-alpha 1.0 \
--train-crop-size 224 --model-ema --val-resize-size 224 --ra-sampler --ra-reps 4 > out.txt 2> err.txt &
