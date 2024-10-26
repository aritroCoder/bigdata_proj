from argparse import ArgumentParser, Namespace
import dataset_prepare

DOMAINS = ['ana', 'awa', 'rae', 'rac', 'rkc', 'rtr', 'rtw']

def generate(params: Namespace):
    args = {
        "dataset": params.dataset,
        "domain": params.domain,
        "type": "iid",
        "fraction": params.fraction,
        "seq_length": params.seq_length,
    }
    args = Namespace(**args)
    dataset_prepare.preprocess(args)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["nab", "ucf101"],
        default="nab",
    )
    parser.add_argument("--domain", type=str, default="ana", choices=DOMAINS)
    parser.add_argument(
        "--fraction", type=float, default=0.8, help="Propotion of train test split"
    )
    parser.add_argument("--seq_length", type=int, default=30, help="Sequence length")
    args = parser.parse_args()
    generate(args)
