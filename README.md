## Set up environment
Create a `.env` file in the root directory with the following content:
```bash
DATASET_DIR=/home/fishnak/Documents/Coding/python/bigdata_proj/dataset
```

## Generate dataset
```bash
python generate_clients_dataset.py --dataset nab --domain rae --fraction 0.8 --seq_length 5
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