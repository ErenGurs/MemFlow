

alias conductor='AWS_EC2_METADATA_DISABLED=true aws --endpoint-url https://conductor.data.apple.com'

conductor s3 cp --recursive s3://egurses-frc/ImagePairs/ ./ImagePairs/
conductor s3 cp s3://mingchen_li/real_validation1.zip ./
# s3zip -e s3://mingchen_li/real_validation1.zip ./test_real/


conda create --name memflow python=3.8
conda activate memflow
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
pip install yacs loguru einops timm==0.4.12 imageio matplotlib tensorboard scipy opencv-python h5py tqdm

