# Copyright 2023 HPC-AI Tech Inc.
# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import time

import fastfold
import numpy as np
import torch
import torch.multiprocessing as mp
from fastfold.config import model_config
from fastfold.data import data_transforms
from fastfold.model.fastnn import set_chunk_size
from fastfold.model.hub import AlphaFold
from fastfold.utils.tensor_utils import tensor_tree_map

# TODO: autochunk insertions
import colossalai
from autochunk.autochunk_codegen import AUTOCHUNK_AVAILABLE
# from colossalai.autochunk.autochunk_codegen import AUTOCHUNK_AVAILABLE
from colossalai.fx.graph_module import ColoGraphModule
from colossalai.fx.passes.meta_info_prop import MetaInfoProp
from testing import free_port
# from colossalai.testing import free_port

if AUTOCHUNK_AVAILABLE:
    # from colossalai.autochunk.autochunk_codegen import AutoChunkCodeGen
    from autochunk.autochunk_codegen import AutoChunkCodeGen
    from colossalai.fx.profiler import MetaTensor
    from colossalai.fx.tracer.experimental import ColoTracer, symbolic_trace
from test_autochunk_evoformer_stack import get_data, get_model

if int(torch.__version__.split(".")[0]) >= 1 and int(torch.__version__.split(".")[1]) > 11:
    torch.backends.cuda.matmul.allow_tf32 = True


def random_template_feats(n_templ, n):
    b = []
    batch = {
        "template_mask": np.random.randint(0, 2, (*b, n_templ)),
        "template_pseudo_beta_mask": np.random.randint(0, 2, (*b, n_templ, n)),
        "template_pseudo_beta": np.random.rand(*b, n_templ, n, 3),
        "template_aatype": np.random.randint(0, 22, (*b, n_templ, n)),
        "template_all_atom_mask": np.random.randint(0, 2, (*b, n_templ, n, 37)),
        "template_all_atom_positions": np.random.rand(*b, n_templ, n, 37, 3) * 10,
        "template_torsion_angles_sin_cos": np.random.rand(*b, n_templ, n, 7, 2),
        "template_alt_torsion_angles_sin_cos": np.random.rand(*b, n_templ, n, 7, 2),
        "template_torsion_angles_mask": np.random.rand(*b, n_templ, n, 7),
    }
    batch = {k: v.astype(np.float32) for k, v in batch.items()}
    batch["template_aatype"] = batch["template_aatype"].astype(np.int64)
    return batch


def random_extra_msa_feats(n_extra, n):
    b = []
    batch = {
        "extra_msa": np.random.randint(0, 22, (*b, n_extra, n)).astype(np.int64),
        "extra_has_deletion": np.random.randint(0, 2, (*b, n_extra, n)).astype(np.float32),
        "extra_deletion_value": np.random.rand(*b, n_extra, n).astype(np.float32),
        "extra_msa_mask": np.random.randint(0, 2, (*b, n_extra, n)).astype(np.float32),
    }
    return batch


def generate_batch(n_res):
    batch = {}
    tf = torch.randint(21, size=(n_res,))
    batch["target_feat"] = torch.nn.functional.one_hot(tf, 22).float()
    batch["aatype"] = torch.argmax(batch["target_feat"], dim=-1)
    batch["residue_index"] = torch.arange(n_res)
    batch["msa_feat"] = torch.rand((128, n_res, 49))
    t_feats = random_template_feats(4, n_res)
    batch.update({k: torch.tensor(v) for k, v in t_feats.items()})
    extra_feats = random_extra_msa_feats(5120, n_res)
    batch.update({k: torch.tensor(v) for k, v in extra_feats.items()})
    batch["msa_mask"] = torch.randint(low=0, high=2, size=(128, n_res)).float()
    batch["seq_mask"] = torch.randint(low=0, high=2, size=(n_res,)).float()
    batch.update(data_transforms.make_atom14_masks(batch))
    batch["no_recycling_iters"] = torch.tensor(2.)

    add_recycling_dims = lambda t: (t.unsqueeze(-1).expand(*t.shape, 3))
    batch = tensor_tree_map(add_recycling_dims, batch)

    return batch


def inference_model(rank, world_size, result_q, batch, args):
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    # init distributed for Dynamic Axial Parallelism
    fastfold.distributed.init_dap()
    torch.cuda.set_device(rank)
    config = model_config(args.model_name)
    if args.chunk_size:
        config.globals.chunk_size = args.chunk_size

    config.globals.inplace = args.inplace
    config.globals.is_multimer = False
    model = AlphaFold(config)

    model = inject_fastnn(model)
    model = model.eval()
    model = model.cuda()

    set_chunk_size(model.globals.chunk_size)

    with torch.no_grad():
        batch = {k: torch.as_tensor(v).cuda() for k, v in batch.items()}
        t = time.perf_counter()
        out = model(batch)
        print(f"Inference time: {time.perf_counter() - t}")
    out = tensor_tree_map(lambda x: np.array(x.cpu()), out)

    result_q.put(out)

    torch.distributed.barrier()
    torch.cuda.synchronize()


def inference_monomer_model(args):
    batch = generate_batch(args.n_res)
    manager = mp.Manager()
    result_q = manager.Queue()
    torch.multiprocessing.spawn(inference_model, nprocs=args.gpus, args=(args.gpus, result_q, batch, args))
    out = result_q.get()

    # get unrelexed pdb and save
    # batch = tensor_tree_map(lambda x: np.array(x[..., -1].cpu()), batch)
    # plddt = out["plddt"]
    # plddt_b_factors = np.repeat(plddt[..., None], residue_constants.atom_type_num, axis=-1)
    # unrelaxed_protein = protein.from_prediction(features=batch,
    #                                             result=out,
    #                                             b_factors=plddt_b_factors)
    # with open('demo_unrelex.pdb', 'w+') as fp:
    #     fp.write(unrelaxed_protein)


def main(args):
    inference_monomer_model(args)

# FASTNN
import torch

from fastfold.model.fastnn import EvoformerStack, ExtraMSAStack
from fastfold.model.fastnn.embedders import TemplateEmbedder
from fastfold.model.fastnn.embedders_multimer import TemplateEmbedderMultimer
from fastfold.model.fastnn.ops import RecyclingEmbedder, InputEmbedder
from fastfold.model.nn.triangular_multiplicative_update import is_fused_triangle_multiplication

def copy_layernorm(model_fast, model_ori):
    model_fast.weight.copy_(model_ori.weight)
    model_fast.bias.copy_(model_ori.bias)


def copy_linear(model_fast, model_ori):
    model_fast.weight.copy_(model_ori.weight)
    if model_fast.use_bias:
        model_fast.bias.copy_(model_ori.bias)


def copy_native_linear(model_fast, model_ori):
    model_fast.weight.copy_(model_ori.weight)
    try:
        model_fast.bias.copy_(model_ori.bias)
    except:
        pass


def copy_kv_linear(model_fast, ori_k, ori_v):
    model_fast.weight.copy_(torch.cat((ori_k.weight, ori_v.weight), dim=0))


def copy_qkv_linear(model_fast, ori_q, ori_k, ori_v):
    model_fast.weight.copy_(torch.cat((ori_q.weight, ori_k.weight, ori_v.weight), dim=0))


def copy_attention(model_fast, model_ori):
    copy_qkv_linear(model_fast.to_qkv, model_ori.linear_q, model_ori.linear_k, model_ori.linear_v)
    copy_linear(model_fast.gating_linear, model_ori.linear_g)
    copy_linear(model_fast.o_linear, model_ori.linear_o)

    try:
        model_fast.gating_bias.copy_(model_ori.linear_g.bias)
    except:
        print("no gating_bias need copy")


def copy_left_right(model_fast, ori_left, ori_right):
    model_fast.weight.copy_(torch.cat((ori_left.weight, ori_right.weight), dim=0))
    model_fast.bias.copy_(torch.cat((ori_left.bias, ori_right.bias), dim=0))


def copy_transition(model_fast, model_ori):
    copy_layernorm(model_fast.norm, model_ori.layer_norm)
    copy_linear(model_fast.linear1, model_ori.linear_1)
    copy_linear(model_fast.linear2, model_ori.linear_2)


def copy_triangle(model_fast, model_ori):
    copy_layernorm(model_fast.layernorm1, model_ori.layer_norm_in)
    copy_layernorm(model_fast.layernorm2, model_ori.layer_norm_out)

    copy_linear(model_fast.output_projection, model_ori.linear_z)
    model_fast.output_bias.copy_(model_ori.linear_z.bias)

    if is_fused_triangle_multiplication():
        copy_linear(model_fast.output_gate, model_ori.linear_gate)
        copy_linear(model_fast.left_right_projection, model_ori.linear_p)
        copy_linear(model_fast.left_right_gate, model_ori.linear_g)
    else:
        copy_linear(model_fast.output_gate, model_ori.linear_g)
        copy_left_right(model_fast.left_right_projection, model_ori.linear_a_p, model_ori.linear_b_p)
        copy_left_right(model_fast.left_right_gate, model_ori.linear_a_g, model_ori.linear_b_g)


def copy_triangle_att(model_fast, model_ori):
    copy_layernorm(model_fast.layernorm1, model_ori.layer_norm)
    copy_linear(model_fast.linear_b, model_ori.linear)
    copy_attention(model_fast.attention, model_ori.mha)

    model_fast.out_bias.copy_(model_ori.mha.linear_o.bias)


def copy_native_att(model_fast, model_ori):
    copy_native_linear(model_fast.linear_q, model_ori.linear_q)
    copy_native_linear(model_fast.linear_k, model_ori.linear_k)
    copy_native_linear(model_fast.linear_v, model_ori.linear_v)
    copy_native_linear(model_fast.linear_o, model_ori.linear_o)
    if model_ori.gating:
         copy_native_linear(model_fast.linear_g, model_ori.linear_g)


def copy_evoformer_para(block_fast, block_ori):
    # msa_stack
    # MSARowAttentionWithPairBias
    copy_layernorm(block_fast.msa.MSARowAttentionWithPairBias.layernormM,
                   block_ori.msa_att_row.layer_norm_m)
    copy_layernorm(block_fast.msa.MSARowAttentionWithPairBias.layernormZ,
                   block_ori.msa_att_row.layer_norm_z)

    copy_attention(block_fast.msa.MSARowAttentionWithPairBias.attention,
                   block_ori.msa_att_row.mha)

    block_fast.msa.MSARowAttentionWithPairBias.linear_b_weights.copy_(
        block_ori.msa_att_row.linear_z.weight)

    block_fast.msa.MSARowAttentionWithPairBias.out_bias.copy_(
        block_ori.msa_att_row.mha.linear_o.bias)

    # MSAColumnAttention
    copy_layernorm(block_fast.msa.MSAColumnAttention.layernormM,
                   block_ori.msa_att_col._msa_att.layer_norm_m)

    copy_attention(block_fast.msa.MSAColumnAttention.attention,
                   block_ori.msa_att_col._msa_att.mha)

    # MSATransition
    copy_transition(block_fast.msa.MSATransition, block_ori.core.msa_transition)

    # communication
    copy_layernorm(block_fast.communication.layernormM,
                   block_ori.core.outer_product_mean.layer_norm)
    copy_linear(block_fast.communication.linear_a, block_ori.core.outer_product_mean.linear_1)
    copy_linear(block_fast.communication.linear_b, block_ori.core.outer_product_mean.linear_2)
    copy_linear(block_fast.communication.o_linear, block_ori.core.outer_product_mean.linear_out)

    # pair_stack
    # TriangleMultiplicationOutgoing
    copy_triangle(block_fast.pair.TriangleMultiplicationOutgoing, block_ori.core.tri_mul_out)
    # TriangleMultiplicationIncoming
    copy_triangle(block_fast.pair.TriangleMultiplicationIncoming, block_ori.core.tri_mul_in)

    # TriangleAttentionStartingNode
    copy_triangle_att(block_fast.pair.TriangleAttentionStartingNode,
                      block_ori.core.tri_att_start)
    copy_triangle_att(block_fast.pair.TriangleAttentionEndingNode, block_ori.core.tri_att_end)

    copy_transition(block_fast.pair.PairTransition, block_ori.core.pair_transition)


def copy_global_attention(model_fast, model_ori):
    copy_linear(model_fast.to_q, model_ori.linear_q)
    copy_kv_linear(model_fast.to_kv, model_ori.linear_k, model_ori.linear_v)
    copy_linear(model_fast.gating_linear, model_ori.linear_g)
    copy_linear(model_fast.o_linear, model_ori.linear_o)

    try:
        model_fast.gating_bias.copy_(model_ori.linear_g.bias)
    except:
        print("no gating_bias need copy")


def copy_extra_msa_para(block_fast, block_ori):
    # msa_stack
    # MSARowAttentionWithPairBias
    copy_layernorm(
        block_fast.msa_stack.MSARowAttentionWithPairBias.layernormM,
        block_ori.msa_att_row.layer_norm_m,
    )
    copy_layernorm(
        block_fast.msa_stack.MSARowAttentionWithPairBias.layernormZ,
        block_ori.msa_att_row.layer_norm_z,
    )

    copy_attention(
        block_fast.msa_stack.MSARowAttentionWithPairBias.attention,
        block_ori.msa_att_row.mha,
    )

    block_fast.msa_stack.MSARowAttentionWithPairBias.linear_b_weights.copy_(
        block_ori.msa_att_row.linear_z.weight
    )

    block_fast.msa_stack.MSARowAttentionWithPairBias.out_bias.copy_(
        block_ori.msa_att_row.mha.linear_o.bias
    )

    # MSAColumnAttention
    copy_layernorm(
        block_fast.msa_stack.MSAColumnAttention.layernormM,
        block_ori.msa_att_col.layer_norm_m,
    )

    copy_global_attention(
        block_fast.msa_stack.MSAColumnAttention.global_attention,
        block_ori.msa_att_col.global_attention,
    )

    # MSATransition
    copy_transition(block_fast.msa_stack.MSATransition, block_ori.core.msa_transition)

    # communication
    comm_model = (
        block_ori.core.outer_product_mean# if not block_ori.is_multimer else block_ori.outer_product_mean
    )
    copy_layernorm(block_fast.communication.layernormM, comm_model.layer_norm)
    copy_linear(block_fast.communication.linear_a, comm_model.linear_1)
    copy_linear(block_fast.communication.linear_b, comm_model.linear_2)
    copy_linear(block_fast.communication.o_linear, comm_model.linear_out)

    # pair_stack
    # TriangleMultiplicationOutgoing
    copy_triangle(
        block_fast.pair_stack.TriangleMultiplicationOutgoing, block_ori.core.tri_mul_out
    )
    # TriangleMultiplicationIncoming
    copy_triangle(
        block_fast.pair_stack.TriangleMultiplicationIncoming, block_ori.core.tri_mul_in
    )

    # TriangleAttentionStartingNode
    copy_triangle_att(
        block_fast.pair_stack.TriangleAttentionStartingNode,
        block_ori.core.tri_att_start,
    )
    copy_triangle_att(
        block_fast.pair_stack.TriangleAttentionEndingNode, block_ori.core.tri_att_end
    )

    copy_transition(
        block_fast.pair_stack.PairTransition, block_ori.core.pair_transition
    )


def copy_template_pair_stack_para(block_fast, block_ori):
    # TriangleMultiplicationOutgoing
    copy_triangle(block_fast.TriangleMultiplicationOutgoing, block_ori.tri_mul_out)
    # TriangleMultiplicationIncoming
    copy_triangle(block_fast.TriangleMultiplicationIncoming, block_ori.tri_mul_in)

    # TriangleAttentionStartingNode
    copy_triangle_att(block_fast.TriangleAttentionStartingNode, block_ori.tri_att_start)
    copy_triangle_att(block_fast.TriangleAttentionEndingNode, block_ori.tri_att_end)

    copy_transition(block_fast.PairTransition, block_ori.pair_transition)


def copy_template_pair_block_para(fast_module, target_module):
    with torch.no_grad():
        for ori_block, fast_block in zip(target_module.blocks, fast_module.blocks):
            copy_template_pair_stack_para(fast_block, ori_block)
            if ori_block.training == False:
                fast_block.eval()


def copy_template_para(block_fast, block_ori):
    # TemplateAngleEmbedder
    copy_linear(block_fast.template_angle_embedder.linear_1,
                block_ori.template_angle_embedder.linear_1)
    copy_linear(block_fast.template_angle_embedder.linear_2,
                block_ori.template_angle_embedder.linear_2)

    # TemplatePairEmbedder
    copy_linear(block_fast.template_pair_embedder.linear,
                block_ori.template_pair_embedder.linear)

    # TemplatePairStack
    copy_template_pair_block_para(block_fast.template_pair_stack,
                                  block_ori.template_pair_stack)
    copy_layernorm(block_fast.template_pair_stack.layer_norm,
                   block_ori.template_pair_stack.layer_norm)

    # TemplatePointwiseAttention
    copy_native_att(block_fast.template_pointwise_att.mha,
                    block_ori.template_pointwise_att.mha)


def copy_template_multimer_para(block_fast, block_ori):
    # TemplatePairEmbedderMultimer
    copy_linear(block_fast.template_pair_embedder.dgram_linear,
                block_ori.template_pair_embedder.dgram_linear)
    copy_linear(block_fast.template_pair_embedder.aatype_linear_1,
                block_ori.template_pair_embedder.aatype_linear_1)
    copy_linear(block_fast.template_pair_embedder.aatype_linear_2,
                block_ori.template_pair_embedder.aatype_linear_2)
    copy_layernorm(block_fast.template_pair_embedder.query_embedding_layer_norm,
                   block_ori.template_pair_embedder.query_embedding_layer_norm)
    copy_linear(block_fast.template_pair_embedder.query_embedding_linear,
                block_ori.template_pair_embedder.query_embedding_linear)
    copy_linear(block_fast.template_pair_embedder.pseudo_beta_mask_linear,
                block_ori.template_pair_embedder.pseudo_beta_mask_linear)
    copy_linear(block_fast.template_pair_embedder.x_linear,
                block_ori.template_pair_embedder.x_linear)
    copy_linear(block_fast.template_pair_embedder.y_linear,
                block_ori.template_pair_embedder.y_linear)
    copy_linear(block_fast.template_pair_embedder.z_linear,
                block_ori.template_pair_embedder.z_linear)
    copy_linear(block_fast.template_pair_embedder.backbone_mask_linear,
                block_ori.template_pair_embedder.backbone_mask_linear)

    # TemplateSingleEmbedderMultimer
    copy_linear(block_fast.template_single_embedder.template_single_embedder,
                block_ori.template_single_embedder.template_single_embedder)
    copy_linear(block_fast.template_single_embedder.template_projector,
                block_ori.template_single_embedder.template_projector)

    # TemplatePairStack
    copy_template_pair_block_para(block_fast.template_pair_stack,
                                  block_ori.template_pair_stack)
    copy_layernorm(block_fast.template_pair_stack.layer_norm,
                   block_ori.template_pair_stack.layer_norm)

    # linear_t
    copy_linear(block_fast.linear_t, block_ori.linear_t)

def inject_autochunk(model):
    print("Entered inject autochunk")
    model = get_model().cpu().eval()
    meta_args, concrete_args = get_data(128, 50)
    if concrete_args is None:
        concrete_args = []

    meta_graph = symbolic_trace(
        model,
        meta_args={k: v.to(torch.device("meta")) for k, v in meta_args},
        concrete_args={k: v for k, v in concrete_args},
    )

    interp = MetaInfoProp(meta_graph)
    meta_tensors = [MetaTensor(i[1], fake_device="cpu") for i in meta_args] + [i[1] for i in concrete_args]

    interp.propagate(*meta_tensors)
    codegen = AutoChunkCodeGen(
        meta_graph,
        max_memory=None,
    )

    graph = ColoTracer().trace(
        model,
        meta_args={k: v.to(torch.device("meta")) for k, v in meta_args},
        concrete_args={k: v for k, v in concrete_args},
    )
    graph.set_codegen(codegen)
    gm = ColoGraphModule(model, graph, ckpt_codegen=False)
    gm.recompile()

    model.evoformer = model


# TODO: change fast module to the gm
def inject_evoformer(model):
    print("Injected fast evoformer")
    with torch.no_grad():
        target_module = model.evoformer
        # TODO: insert autochunkcode gen here by modifying fast_module
        # Do this before converting to the fast_fold

        model = target_module.cpu().eval()
        meta_args, concrete_args = get_data(1, 1)
        if concrete_args is None:
            concrete_args = []
        meta_graph = symbolic_trace(
            model,
            meta_args={k: v.to(torch.device("meta")) for k, v in meta_args},
            concrete_args={k: v for k, v in concrete_args},
        )
        interp = MetaInfoProp(meta_graph)
        meta_tensors = [MetaTensor(i[1], fake_device="cpu") for i in meta_args] + [i[1] for i in concrete_args]

        print(f"\nfastfold's interp\n")
        print(interp)

        print(f"\nfastfold's meta_tensors\n")
        print(meta_tensors)

        # nterp.propagate(*meta_tensors)

        # fast_module = EvoformerStack(
        #     c_m=target_module.blocks[0].msa_att_row.c_in,
        #     c_z=target_module.blocks[0].msa_att_row.c_z,
        #     c_s=target_module.linear.out_features,
        #     no_blocks=len(target_module.blocks),
        #     blocks_per_ckpt=target_module.blocks_per_ckpt,
        #     clear_cache_between_blocks=target_module.clear_cache_between_blocks,
        #     is_multimer=target_module.blocks[0].is_multimer,
        # )
        # # copy parameters from target block (original blocks) to the fast blocks
        # for target_block, fast_block in zip(target_module.blocks, fast_module.blocks):
        #     copy_evoformer_para(fast_block, target_block)
        #     if target_block.training == False:
        #         fast_block.eval()
        # copy_linear(fast_module.linear, target_module.linear)
        model.evoformer = target_module


def inject_extramsa(model):
    with torch.no_grad():
        target_module = model.extra_msa_stack
        fast_module = ExtraMSAStack(
            c_m=target_module.blocks[0].msa_att_row.c_in,
            c_z=target_module.blocks[0].msa_att_row.c_z,
            no_blocks=len(target_module.blocks),
            clear_cache_between_blocks=target_module.clear_cache_between_blocks,
            is_multimer=target_module.blocks[0].is_multimer,
        )
        for target_block, fast_block in zip(target_module.blocks, fast_module.blocks):
            copy_extra_msa_para(fast_block, target_block)
            if target_block.training == False:
                fast_block.eval()
        model.extra_msa_stack = fast_module


def inject_template(model):
    with torch.no_grad():
        if model.evoformer.blocks[0].is_multimer:
            target_module = model.template_embedder
            fast_module = TemplateEmbedderMultimer(config=model.template_embedder.config)
            copy_template_multimer_para(fast_module, target_module)
            if target_module.training == False:
                fast_module.eval()
            model.template_embedder = fast_module
        else:
            target_module = model.template_embedder
            fast_module = TemplateEmbedder(config=model.template_embedder.config)
            copy_template_para(fast_module, target_module)
            if target_module.training == False:
                fast_module.eval()
            model.template_embedder = fast_module


def inject_embedder(model):
    if model.evoformer.blocks[0].is_multimer:
        return

    # recycle embedder
    with torch.no_grad():
        target_module = model.recycling_embedder
        fast_module = RecyclingEmbedder(
            c_m=target_module.c_m,
            c_z=target_module.c_z,
            min_bin=target_module.min_bin,
            max_bin=target_module.max_bin,
            no_bins=target_module.no_bins,
            inf=target_module.inf
        )
        copy_native_linear(fast_module.linear, target_module.linear)
        copy_layernorm(fast_module.layer_norm_m, target_module.layer_norm_m)
        copy_layernorm(fast_module.layer_norm_z, target_module.layer_norm_z)
        if target_module.training == False:
            fast_module.eval()
        model.recycling_embedder = fast_module

    # input embedder
    with torch.no_grad():
        target_module = model.input_embedder
        fast_module = InputEmbedder(
            tf_dim=target_module.tf_dim,
            msa_dim=target_module.msa_dim,
            c_z=target_module.c_z,
            c_m=target_module.c_m,
            relpos_k=target_module.relpos_k,
        )
        copy_linear(fast_module.linear_tf_z_i, target_module.linear_tf_z_i)
        copy_linear(fast_module.linear_tf_z_j, target_module.linear_tf_z_j)
        copy_linear(fast_module.linear_tf_m, target_module.linear_tf_m)
        copy_linear(fast_module.linear_msa_m, target_module.linear_msa_m)
        copy_linear(fast_module.linear_relpos, target_module.linear_relpos)
        if target_module.training == False:
            fast_module.eval()
        model.input_embedder = fast_module


def inject_fastnn(model):
    inject_autochunk(model)
    # inject_evoformer(model)
    inject_extramsa(model)
    inject_template(model)
    inject_embedder(model)
    return model
# FASTNN

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, default=1, help="""Number of GPUs with which to run inference""")
    parser.add_argument("--n_res", type=int, default=50, help="virtual residue number of random data")
    parser.add_argument("--model_name", type=str, default="model_1", help="model name of alphafold")
    parser.add_argument('--chunk_size', type=int, default=None)
    parser.add_argument('--inplace', default=False, action='store_true')

    args = parser.parse_args()

    main(args)