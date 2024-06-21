from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from contextlib import contextmanager
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tensor import pad_to_length
from .utils import unwrap_model


@dataclass
class DPOConfig:
    ref_model: Optional[nn.Module] = None
    use_peft: bool = False
    pad_token_id: int = 0
    ignore_index: int = -100
    loss_type: str = "sigmoid"
    beta: float = 0.01
    label_smoothing: float = 0.0
    reference_free: bool = False


def create_reference_model(model):
    """
    根据model创建一个reference model，深拷贝，requires_grad=False，ref_model.eval()
    """

    parameter_names = [n for n, _ in model.named_parameters()]
    ref_model = copy.deepcopy(model)

    # if no layers are shared, return copy of model
    for param_name in parameter_names:
        param = ref_model.get_parameter(param_name)
        param.requires_grad = False
        
    return ref_model.eval()


def concatenated_inputs(
    batch: Dict[str, Union[List, torch.LongTensor]],
    dpo_config,
) -> Dict[str, torch.LongTensor]:
    """Concatenate the chosen and rejected inputs into a single tensor.
    
    将batch拼接，即chosen和rejected在batch维度上拼接，并行计算
    此时，batch_size维度上，[:len_chosen]是chosen，[len_chosen:]是rejected

    Args:
        batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).
        pad_token_id: The padding value to use for the concatenated inputs_ids.
        ignore_index: The label pad token id.
        device: The device for the concatenated inputs.

    Returns:
        A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
    """
    device = batch["chosen_input_ids"].device
    ignore_index = dpo_config.ignore_index
    pad_token_id = dpo_config.pad_token_id
    
    concatenated_batch = {}

    max_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])

    # 先将chosen和rejected进行pad，确保长度一致
    # chosen
    for k in batch:
        if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
            # 确定pad的值
            if "labels" in k:
                pad_value = ignore_index
            elif k.endswith("_input_ids"):
                pad_value = pad_token_id
            elif k.endswith("_attention_mask"):
                pad_value = 0
                
            # 添加到concatenated_batch中
            concatenated_key = k.replace("chosen", "concatenated")
            concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
    # rejected
    for k in batch:
        if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
            # 确定pad的值
            if "labels" in k:
                pad_value = ignore_index
            elif k.endswith("_input_ids"):
                pad_value = pad_token_id
            elif k.endswith("_attention_mask"):
                pad_value = 0
                
            # pad到一样的长度，然后进行拼接
            concatenated_key = k.replace("rejected", "concatenated")
            concatenated_batch[concatenated_key] = torch.cat(
                (
                    concatenated_batch[concatenated_key],
                    pad_to_length(batch[k], max_length, pad_value=pad_value),
                ),
                dim=0,
            ).to(device=device)

    return concatenated_batch


def concatenated_forward(
    model: nn.Module, concatenated_batch: Dict[str, Union[List, torch.LongTensor]], len_chosen, loss_mask, dpo_config
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

    We do this to avoid doing two forward passes, because it's faster for FSDP.
    
    使用concatenated_inputs后的batch进行forward
    此时，batch_size维度上，[:len_chosen]是chosen，[len_chosen:]是rejected
    """
    ignore_index = dpo_config.ignore_index
    loss_type = dpo_config.loss_type
    
    labels = concatenated_batch["concatenated_labels"]

    # 前向计算
    all_logits = model(
        concatenated_batch["concatenated_input_ids"],
        attention_mask=concatenated_batch["concatenated_attention_mask"],
        use_cache=False,
    )["logits"]
    
    # 手动计算损失
    # 将ignore_index的label设置为0，确保不会越界，后面会通过loss_mask找到这些位置，并且掩码
    # 防止原地操作
    labels = labels.clone()
    labels[labels == ignore_index] = 0
    per_token_logps = torch.gather(all_logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    all_logps, size_completion = (per_token_logps * loss_mask).sum(-1), loss_mask.sum(-1)

    chosen_logps_avg = all_logps[:len_chosen] / size_completion[:len_chosen]

    if loss_type == "ipo":
        all_logps = all_logps / size_completion

    chosen_logps = all_logps[:len_chosen]
    rejected_logps = all_logps[len_chosen:]

    chosen_logits = all_logits[:len_chosen]
    rejected_logits = all_logits[len_chosen:]

    # 只有chosen_logps和rejected_logps用于后面的计算
    return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_logps_avg)


def dpo_loss(
    policy_chosen_logps: torch.FloatTensor,
    policy_rejected_logps: torch.FloatTensor,
    reference_chosen_logps: torch.FloatTensor,
    reference_rejected_logps: torch.FloatTensor,
    dpo_config: DPOConfig,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the DPO loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the DPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    """
    device = policy_chosen_logps.device
    
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    if dpo_config.reference_free:
        ref_logratios = torch.tensor([0], dtype=pi_logratios.dtype, device=pi_logratios.device)
    else:
        ref_logratios = reference_chosen_logps - reference_rejected_logps

    pi_logratios = pi_logratios.to(device)
    ref_logratios = ref_logratios.to(device)
    logits = pi_logratios - ref_logratios

    # The beta is a temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5.
    # We ignore the reference model as beta -> 0. The label_smoothing parameter encodes our uncertainty about the labels and
    # calculates a conservative DPO loss.
    if dpo_config.loss_type == "sigmoid":
        losses = (
            -F.logsigmoid(dpo_config.beta * logits) * (1 - dpo_config.label_smoothing)
            - F.logsigmoid(-dpo_config.beta * logits) * dpo_config.label_smoothing
        )
    elif dpo_config.loss_type == "robust":
        losses = (
            -F.logsigmoid(dpo_config.beta * logits) * (1 - dpo_config.label_smoothing)
            + F.logsigmoid(-dpo_config.beta * logits) * dpo_config.label_smoothing
        ) / (1 - 2 * dpo_config.label_smoothing)
    elif dpo_config.loss_type == "hinge":
        losses = torch.relu(1 - dpo_config.beta * logits)
    elif dpo_config.loss_type == "ipo":
        # eqn (17) of the paper where beta is the regularization parameter for the IPO loss, denoted by tau in the paper.
        losses = (logits - 1 / (2 * dpo_config.beta)) ** 2
    elif dpo_config.loss_type == "kto_pair":
        # eqn (7) of the HALOs paper
        chosen_KL = (policy_chosen_logps - reference_chosen_logps).mean().clamp(min=0)
        rejected_KL = (policy_rejected_logps - reference_rejected_logps).mean().clamp(min=0)

        chosen_logratios = policy_chosen_logps - reference_chosen_logps
        rejected_logratios = policy_rejected_logps - reference_rejected_logps
        # As described in the KTO report, the KL term for chosen (rejected) is estimated using the rejected (chosen) half.
        losses = torch.cat(
            (
                1 - F.sigmoid(dpo_config.beta * (chosen_logratios - rejected_KL)),
                1 - F.sigmoid(dpo_config.beta * (chosen_KL - rejected_logratios)),
            ),
            0,
        )
    # elif dpo_config.loss_type == "bco_pair":
    #     chosen_logratios = policy_chosen_logps - reference_chosen_logps
    #     rejected_logratios = policy_rejected_logps - reference_rejected_logps

    #     chosen_rewards = dpo_config.beta * chosen_logratios
    #     rejected_rewards = dpo_config.beta * rejected_logratios
    #     rewards = torch.cat((chosen_rewards, rejected_rewards), 0).mean().detach()
    #     running.update(rewards)
    #     delta = running.mean

    #     losses = -F.logsigmoid((dpo_config.beta * chosen_logratios) - delta) - F.logsigmoid(
    #         -(dpo_config.beta * rejected_logratios - delta)
    #     )
    elif dpo_config.loss_type == "sppo_hard":
        # In the paper (https://arxiv.org/pdf/2405.00675), SPPO employs a soft probability approach, estimated using the PairRM score. The probability calculation is conducted outside of the trainer class. The version described here is the hard probability version, where P in Equation (4.7) of Algorithm 1 is set to 1 for the winner and 0 for the loser.
        a = policy_chosen_logps - reference_chosen_logps
        b = policy_rejected_logps - reference_rejected_logps

        losses = (a - 0.5 / dpo_config.beta) ** 2 + (b + 0.5 / dpo_config.beta) ** 2
    elif dpo_config.loss_type == "nca_pair":
        chosen_rewards = (policy_chosen_logps - reference_chosen_logps) * dpo_config.beta
        rejected_rewards = (policy_rejected_logps - reference_rejected_logps) * dpo_config.beta
        losses = (
            -F.logsigmoid(chosen_rewards)
            - 0.5 * F.logsigmoid(-chosen_rewards)
            - 0.5 * F.logsigmoid(-rejected_rewards)
        )
    else:
        raise ValueError(
            f"Unknown loss type: {dpo_config.loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo', 'kto_pair', 'bco_pair', 'sppo_hard', 'nca_pair', 'robust']"
        )

    chosen_rewards = (
        dpo_config.beta
        * (
            policy_chosen_logps.to(device) - reference_chosen_logps.to(device)
        ).detach()
    )
    rejected_rewards = (
        dpo_config.beta
        * (
            policy_rejected_logps.to(device)
            - reference_rejected_logps.to(device)
        ).detach()
    )

    return losses, chosen_rewards, rejected_rewards


@contextmanager
def null_ref_context(model):
    """Context manager for handling null reference model (that is, peft adapter manipulation)."""
    unwrapped_model = unwrap_model(model)
    unwrapped_model.disable_adapter_layers()
    yield
    unwrapped_model.enable_adapter_layers()


def dpo_forward(model, batch, dpo_config: DPOConfig):
    """
    DPO的前向计算
    """
    ref_model = dpo_config.ref_model
    
    # 1. 将batch拼接，即chosen和rejected在batch维度上拼接，并行计算
    # 此时，batch_size维度上，[:len_chosen]是chosen，[len_chosen:]是rejected
    concatenated_batch = concatenated_inputs(
        batch,
        dpo_config=dpo_config,
    )
    len_chosen = batch["chosen_labels"].shape[0]
    
    # 找出要掩码的部分，只计算非mask的token的损失
    loss_mask = concatenated_batch["concatenated_labels"] != dpo_config.ignore_index

    # 2. policy_model的前向计算
    (
        policy_chosen_logps,
        policy_rejected_logps,
        policy_chosen_logits,
        policy_rejected_logits,
        policy_chosen_logps_avg,
    ) = concatenated_forward(model, concatenated_batch, len_chosen, loss_mask, dpo_config=dpo_config)

    # 3. reference_model的前向计算
    # if reference_chosen_logps and reference_rejected_logps in batch use them, otherwise use the reference model
    # 如果已经提供，则直接使用，否则使用ref_model进行计算
    if (
        "reference_chosen_logps" in batch
        and "reference_rejected_logps" in batch
        # and args.rpo_alpha is not None
    ):
        reference_chosen_logps = batch["reference_chosen_logps"]
        reference_rejected_logps = batch["reference_rejected_logps"]
    else:
        with torch.no_grad():
            if ref_model is None:
                # 将peft禁用，则使用base_model进行计算，这样可以节省一个model的内存
                assert dpo_config.use_peft is True, "ref_model is required if use_peft is False"
                with null_ref_context(model):
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        _,
                        _,
                        _,
                    ) = concatenated_forward(model, concatenated_batch, len_chosen, loss_mask, dpo_config=dpo_config)
            else:
                (
                    reference_chosen_logps,
                    reference_rejected_logps,
                    _,
                    _,
                    _,
                ) = concatenated_forward(ref_model, concatenated_batch, len_chosen, loss_mask, dpo_config=dpo_config)

    # 4. 计算dpo损失
    losses, chosen_rewards, rejected_rewards = dpo_loss(
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
        dpo_config=dpo_config,
    )
    reward_accuracies = (chosen_rewards > rejected_rewards).float()

    # if args.rpo_alpha is not None:
    #     losses = losses * args.rpo_alpha - policy_chosen_logps_avg

    # 指标
    # prefix = "eval_" if train_eval == "eval" else ""
    prefix = ""
    metrics = {}
    metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().cpu()
    metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().cpu()
    metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu()
    metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().cpu()
    metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().mean().cpu()
    metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().mean().cpu()
    metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().mean().cpu()
    metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().mean().cpu()

    return losses.mean(), metrics
    