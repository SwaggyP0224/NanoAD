# NanoAD

A Low-Latency Framework for Industrial Surface Anomaly Detection in Resource-constrained Environments.

## Steps

Clone this repository:

```bash
git clone https://github.com/SwaggyP0224/NanoAD.git
cd NanoAD

pip install -r requirements.txt

Dataset
Please prepare the datasets before training:

MVTec AD dataset
DTD dataset for anomaly source images

Training
Run the following command to train NanoAD:

python main.py \
  --dataset 'MVTec AD' \
  --data_path 'mvtec' \
  --save_dir './results' \
  --batch_size 32 \
  --image_size 288 \
  --backbone 'resnet18' \
  --layers_to_extract 'layer2' 'layer3' \
  --pretrain_embed_dimension 384 \
  --target_embed_dimension 1024 \
  --pre_proj 1 \
  --dsc_layers 2 \
  --dsc_hidden 512 \
  --anomaly_source_path 'dtd/images' \
  --epochs 200 \
  --lr 1e-4
