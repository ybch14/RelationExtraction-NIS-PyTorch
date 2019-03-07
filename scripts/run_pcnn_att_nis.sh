python3.6 train.py --model pcnn_att_nis\
                   --data_path data/processed\
                   --output_path results\
                   --filters 230\
                   --kernel_size 3\
                   --class_num 53\
                   --seq_len 80\
                   --batch_size 160\
                   --epochs 20\
                   --nis_hidden_dims "512, 256, 128, 64"\
