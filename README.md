## Set up environment
Create a `.env` file in the root directory with the following content (replace path your own):
```bash
DATASET_DIR=/home/fishnak/Documents/Coding/python/bigdata_proj/dataset
```

## Generate dataset
```bash
python generate_clients_dataset.py --dataset nab --domain rae --fraction 0.8 --seq_length 30
```
### Arguments
- `--dataset`: dataset name (default: `nab`). Available options:
	- `nab`: Numenta Anomaly Benchmark dataset
	- `ucf101`: Not implemented
- `--domain`: domain name (default: `ana`). Available options:
	- `ana`: Artificial No Anomaly
	- `awa`: Artificial With Anomaly
	- `rae`: Real Ad Exchange
	- `rac`: Real AWS Cloudwatch
	- `rkc`: Real Known Cause
	- `rtr`: Real Traffic
	- `rtw`: Real Tweets
- `--fraction`: fraction of the dataset to be used for train(default: `0.8`)
- `--seq_length`: sequence length for GRU (default: `30`)

## Run Model
```bash
python main.py --dataset nab --trainer fedavg --type rae_s_5_f_8 --model grunet --rounds 5
```

## UCF101 Dataset
```bash
cd UCF101/
mkdir data & cd data/
bash download_ucf101.sh # Downloads the UCF-101 dataset (~7.2 GB)
python extract_frames.py # Extracts frames from the video (~26.2 GB)
python3 train_fl.py
```