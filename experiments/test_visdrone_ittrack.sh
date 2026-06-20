cd src
python3 track.py --load_model '../exp/mot/ittrack_visdrone/model_last.pth' \
                 --test_visdrone True \
                 --data_dir '../data' \
                 --gpus '0'
cd ..
