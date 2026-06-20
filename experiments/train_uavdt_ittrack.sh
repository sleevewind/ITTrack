cd src
python3 train.py --task mot \
                 --exp_id 'ittrack_uavdt' \
                 --batch_size 8 \
                 --load_model '../models/ctdet_coco_dla_2x.pth' \
                 --data_cfg '../src/lib/cfg/uavdt.json' \
                 --data_dir '../data' \
                 --gpus '0,1' \
                 --lr_step '20' \
                 --lr 7e-5 \
                 --num_epochs 30
cd ..
