python3.6 train.py --model apcnn\
                   --data_path data/processed\
                   --output_path results\
                   --filters 230\
                   --kernel_size 3\
                   --class_num 53\
                   --seq_len 80\
                   --batch_size 100\
                   --epochs 20\
                   --slice_rate 0\
