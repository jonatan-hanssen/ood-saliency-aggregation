import torch
import torch.nn as nn
from tqdm import tqdm

from typing import Callable, Any
from torch.utils.data import DataLoader

from openood.postprocessors.base_postprocessor import BasePostprocessor


class SalAggPlusMLS(BasePostprocessor):
    def __init__(
        self,
        config,
        saliency_generator: Callable[[torch.Tensor], torch.Tensor],
        aggregator: Callable[[torch.Tensor], torch.Tensor],
        device_str: str = 'cuda',
    ):
        super().__init__(config)
        self.setup_flag = False
        self.APS_mode = False
        self.saliency_generator = saliency_generator
        self.aggregator = aggregator
        self.device = torch.device(device_str)

        self.logit_std = None
        self.saliency_std = None
        self.sign = None

    def setup(
        self,
        net: nn.Module,
        id_loader_dict: dict[str, DataLoader],
        ood_loader_dict: dict[str, DataLoader],
    ):
        if not self.setup_flag:
            net.eval()

            all_max_logits = list()
            all_saliency_aggregates = list()

            for batch in tqdm(
                id_loader_dict['val'], desc='ID Setup: ', position=0, leave=True
            ):
                data = batch['data'].to(self.device)

                max_logits, _ = torch.max(net(data), dim=-1)
                saliencies = self.saliency_generator(data)
                saliencies = saliencies.reshape(
                    saliencies.shape[0],
                    int(torch.prod(torch.tensor(saliencies.shape[1:]))),
                )

                aggregate = self.aggregator(saliencies)

                max_logits = max_logits.detach().cpu()

                all_max_logits.append(max_logits)
                all_saliency_aggregates.append(aggregate)

            self.logit_std = torch.cat(all_max_logits).std()
            self.saliency_std = torch.cat(all_saliency_aggregates).std()

            id_saliency_mean = torch.cat(all_saliency_aggregates).mean()

            all_saliency_aggregates = list()
            for batch in tqdm(
                ood_loader_dict['val'], desc='OOD Setup: ', position=0, leave=True
            ):
                data = batch['data'].to(self.device)

                saliencies = self.saliency_generator(data)
                saliencies = saliencies.reshape(
                    saliencies.shape[0],
                    int(torch.prod(torch.tensor(saliencies.shape[1:]))),
                )

                aggregate = self.aggregator(saliencies)

                all_saliency_aggregates.append(aggregate)

            ood_saliency_mean = torch.cat(all_saliency_aggregates).mean()

            self.sign = 1 if id_saliency_mean > ood_saliency_mean else -1
            self.setup_flag = True
        else:
            pass

    def postprocess(self, net: nn.Module, data: Any):
        assert self.logit_std is not None
        assert self.saliency_std is not None
        assert self.sign is not None

        data = data.to(self.device)
        max_logits, preds = torch.max(net(data), dim=-1)

        saliencies = self.saliency_generator(data)
        saliencies = saliencies.reshape(
            saliencies.shape[0],
            int(torch.prod(torch.tensor(saliencies.shape[1:]))),
        )

        aggregate = self.aggregator(saliencies)

        max_logits = max_logits.detach().cpu()
        preds = preds.detach().cpu()

        score_ood = max_logits / self.logit_std + self.sign * (
            aggregate / self.saliency_std
        )
        return preds, score_ood
