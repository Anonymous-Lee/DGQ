
datapath=/CIR_Datasets/FashionIQ

python main.py --dataset fashioniq_shirt --split val  --dataset-path $datapath --preload img_features captions mods --clip ViT-B-32 --blip o4-mini
python main.py --dataset fashioniq_shirt --split val  --dataset-path $datapath --preload img_features captions mods --clip ViT-L-14 --blip o4-mini