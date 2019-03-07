mkdir data
cp data.zip data
cd data/
unzip data.zip
rm data.zip
cd ../
python3.6 preprocess.py --data_path data --filter_h 3 --max_len 80 --output_path data/processed
