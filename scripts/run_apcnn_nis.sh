python3.6 train.py --model apcnn_nis\
                   --data_path data/processed\
                   --output_path results\
                   --filters 230\
                   --kernel_size 3\
                   --class_num 53\
                   --seq_len 80\
                   --batch_size 100\
                   --epochs 1\
                   --nis_hidden_dims "512, 256, 128, 64"\
