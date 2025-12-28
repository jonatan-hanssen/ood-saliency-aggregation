from src.salagg_plus_mls import SalAggPlusMLS
from src.utils import get_network, get_aggregate_function, get_saliency_generator
import argparse

from openood.evaluation_api import Evaluator


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', '-d', type=str, default='cifar100')
    parser.add_argument('--saliency_generator', '-s', type=str, default='gbp')
    parser.add_argument('--batch_size', '-b', type=int, default=200)
    parser.add_argument('--aggregator', '-a', type=str, default='norm')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    network = get_network(args.dataset, args.device)

    postprocessor = SalAggPlusMLS(
        config=None,
        saliency_generator=get_saliency_generator(args.saliency_generator, network),
        aggregator=get_aggregate_function(args.aggregator),
        device_str=args.device,
    )

    evaluator = Evaluator(
        net=network,
        id_name=args.dataset,
        data_root='./data',
        config_root=None,
        postprocessor=postprocessor,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
    )

    metrics = evaluator.eval_ood(fsood=False)
    print(metrics)


if __name__ == '__main__':
    main()
