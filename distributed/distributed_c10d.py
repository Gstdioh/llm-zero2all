import torch
import torch.distributed


def get_global_rank(group: torch.distributed.ProcessGroup, group_rank: int) -> int:
    """
    Translate a group rank into a global rank.

    ``group_rank`` must be part of `group` otherwise this raises RuntimeError.

    Args:
        group (ProcessGroup): ProcessGroup to find the global rank from.
        group_rank (int): Group rank to query.

    Returns:
        Global rank of ``group_rank`` relative to ``group``

    N.B. calling this function on the default process group returns identity
    
    pytorch1.12中没有这个函数，我自己添加一下
    """
    if group is torch.distributed.GroupMember.WORLD:
        return group_rank
    if group not in torch.distributed.distributed_c10d._pg_group_ranks:
        raise ValueError(f"Group {group} is not registered, please create group with torch.distributed.new_group API")
    for rank, grp_rank in torch.distributed.distributed_c10d._pg_group_ranks[group].items():
        if grp_rank == group_rank:
            return rank
    raise ValueError(f"Group rank {group_rank} is not part of group {group}")
