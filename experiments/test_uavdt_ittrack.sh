cd src
python3 track.py --load_model '../exp/mot/ittrack_uavdt/model_last.pth' \
                 --test_uavdt True \
                 --data_dir '../data' \
                 --gpus '0,1'
cd ..
