from typing import List, Tuple

import pytest
import torch
import torch.fx

try:
    from fastfold.model.nn.evoformer import EvoformerStack

    HAS_REPO = True
except:
    HAS_REPO = False

from test_autochunk_alphafold_utils import run_test

# from colossalai.autochunk.autochunk_codegen import AUTOCHUNK_AVAILABLE
from autochunk.autochunk_codegen import AUTOCHUNK_AVAILABLE
from testing import clear_cache_before_run, parameterize, spawn
# from colossalai.testing import clear_cache_before_run, parameterize, spawn

#TODO: inject fast nn!
def get_model():
    model = (
        EvoformerStack(
            c_m=256,
            c_z=128,
            c_hidden_msa_att=32,
            c_hidden_opm=32,
            c_hidden_mul=128,
            c_hidden_pair_att=32,
            c_s=384,
            no_heads_msa=8,
            no_heads_pair=4,
            no_blocks=48,  # 48
            transition_n=4,
            msa_dropout=0.15,
            pair_dropout=0.25,
            blocks_per_ckpt=None,
            inf=1000000000.0,
            eps=1e-08,
            clear_cache_between_blocks=False,
            is_multimer=False,
        )
        .eval()
        .cuda()
    )
    return model


def get_data(msa_len: int, pair_len: int) -> Tuple[List, List]:
    node = torch.randn(1, msa_len, pair_len, 256).cuda()
    node_mask = torch.randn(1, msa_len, pair_len).cuda()
    pair = torch.randn(1, pair_len, pair_len, 128).cuda()
    pair_mask = torch.randn(1, pair_len, pair_len).cuda()

    # TODO: changed "msa_mask" to "seq_mask"
    # c_m = 256, c_z = 128
    meta_args = [
        ("m", node),
        ("z", pair),
        ("msa_mask", node_mask),
        ("pair_mask", pair_mask),
    ]
    concrete_args = [("chunk_size", None), ("_mask_trans", True)]
    # print("\nmeta_args node\n")
    # print(meta_args[0][1])
    return meta_args, concrete_args


@pytest.mark.skipif(
    not (AUTOCHUNK_AVAILABLE and HAS_REPO),
    reason="torch version is lower than 1.12.0",
)
@clear_cache_before_run()
@parameterize("max_memory", [None, 20, 24])
@parameterize("data_args", [(32, 64)])  # (msa_len, pair_len)
def test_evoformer_stack(data_args, max_memory):
    spawn(
        run_test,
        1,
        data_args=data_args,
        max_memory=max_memory,
        get_model=get_model,
        get_data=get_data,
    )


if __name__ == "__main__":
    run_test(
        rank=0,
        data_args=(32, 64),
        max_memory=None,
        get_model=get_model,
        get_data=get_data,
        print_code=True,
        print_mem=True,
        print_progress=True,
    )
