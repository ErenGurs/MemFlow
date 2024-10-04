

mkdir ckpts/
wget -P ckpts/ https://github.com/DQiaole/MemFlow/releases/download/v1.0.0/MemFlowNet_sintel.pth
wget -P ckpts/ https://github.com/DQiaole/MemFlow/releases/download/v1.0.0/MemFlowNet_things.pth

alias conductor='AWS_EC2_METADATA_DISABLED=true aws --endpoint-url https://conductor.data.apple.com'

# conductor s3 cp --recursive s3://egurses-frc/ImagePairs/ ./ImagePairs/
conductor s3 cp s3://mingchen_li/real_validation1.zip ./
# s3zip -e s3://mingchen_li/real_validation1.zip ./test_real/


# Original conda env creation takes very long and some issues with resolving.
# Therefore I found out, the requirements are quite compatible with the conda iris env
# that comes with the iris docker. So I simply clone it as "memflow" env.
#
#conda create --name memflow python=3.8
#conda activate memflow
#conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
#pip install yacs loguru einops timm==0.4.12 imageio matplotlib tensorboard scipy opencv-python h5py tqdm
#
conda create --name memflow --clone iris
conda activate memflow

pip install yacs loguru einops timm==0.4.12 # imageio matplotlib tensorboard scipy opencv-python h5py tqdm


# To be removed later:
# An experiment to run MemFlow on sequence data
conductor s3 cp --recursive s3://egurses-frc/BMS/input/bms.video.new/ ./bms.video.new/
mkdir val5_holes_new/
ffmpeg -i bms.video.new/715G3Q7O2IS9C8NJ_IMG_0073.MOV  val5_holes_new/'%04d.png'
inference.py --name MemFlowNet --stage sintel --restore_ckpt ckpts/MemFlowNet_things.pth --seq_dir ./val5_holes_new/ --vis_dir temp/val5_holes_new/