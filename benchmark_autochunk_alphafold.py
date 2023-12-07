import time
from typing import Any

import torch
import torch.fx

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


def _benchmark_evoformer_stack_gm(
    data_args: tuple,
    max_memory: int,
    get_model: Any,
    get_data: Any,
) -> None:
    # build model and input
    model = get_model().cpu().eval()
    meta_args, concrete_args = get_data(1, 1)
    if concrete_args is None:
        concrete_args = []

    # trace the meta graph and setup codegen
    # meta_graph = symbolic_trace(
    #     model,
    #     meta_args={k: v.to(torch.device("meta")) for k, v in meta_args},
    #     concrete_args={k: v for k, v in concrete_args},
    # )

    # TODO: removed meta_args and concrete_args
    meta_graph = symbolic_trace(
        model,
        meta_args={k: v.to(torch.device("meta")) for k, v in meta_args},
        concrete_args={k: v for k, v in concrete_args},
    )

    interp = MetaInfoProp(meta_graph)
    meta_tensors = [MetaTensor(i[1], fake_device="cpu") for i in meta_args] + [i[1] for i in concrete_args]

    print(f"\nalphafold's interp\n")
    print(interp)

    print(f"\nalphafold's meta_tensors\n")
    print(meta_tensors)
    
    interp.propagate(*meta_tensors)
    codegen = AutoChunkCodeGen(
        meta_graph,
        max_memory=max_memory,
    )

    # trace and recompile
    # MetaInfoProp requires symbolic_trace but CodeGen requires ColoTracer
    graph = ColoTracer().trace(
        model,
        meta_args={k: v.to(torch.device("meta")) for k, v in meta_args},
        concrete_args={k: v for k, v in concrete_args},
    )
    graph.set_codegen(codegen)
    gm = ColoGraphModule(model, graph, ckpt_codegen=False)
    gm.recompile()

    # init inputs
    inputs = [i[1] for i in meta_args] + [i[1] for i in concrete_args]
    inputs = [i.cuda() if isinstance(i, torch.Tensor) else i for i in inputs]
    model.cuda()

    # print(f"\nmodel: {model}\n")
    # bench
    mem = _benchmark_memory(gm, inputs)
    speed = _benchmark_speed(gm, inputs)
    print("evoformer stack autochunk, peak mem: %.2fMB, time: %.4fs" % (mem, speed))


def _benchmark_evoformer_stack_origin(
    data_args: tuple,
    get_model: Any,
    get_data: Any,
) -> None:
    # build model and input
    model = get_model()
    meta_args, concrete_args = get_data(*data_args)
    if concrete_args is None:
        concrete_args = []

    # init inputs
    inputs = [i[1] for i in meta_args] + [i[1] for i in concrete_args]
    inputs = [i.cuda() if isinstance(i, torch.Tensor) else i for i in inputs]
    model.cuda()

    # bench
    mem = _benchmark_memory(model, inputs)
    speed = _benchmark_speed(model, inputs)
    print("evoformer stack origin, peak mem: %.2fMB, time: %.4fs" % (mem, speed))
    return mem


def _benchmark_memory(model, inputs):
    with torch.no_grad():
        torch.cuda.reset_peak_memory_stats()
        now_mem = torch.cuda.memory_allocated() / 1024**2
        # print("\ninputs\n")
        # print(*inputs)
        model(*inputs)
        new_max_mem = torch.cuda.max_memory_allocated() / 1024**2
    return new_max_mem - now_mem


def _benchmark_speed(model, inputs, loop=5):
    with torch.no_grad():
        for _ in range(loop // 2 + 1):
            model(*inputs)
        torch.cuda.synchronize()
        time1 = time.time()
        for _ in range(loop):
            model(*inputs)
        torch.cuda.synchronize()
        time2 = time.time()
    return (time2 - time1) / loop


def benchmark_evoformer_stack(data_args):
    from test_autochunk_evoformer_stack import get_data, get_model

    # Origin, Autochunk high budget -> low budget; NOne constraints
    # print("\nmsa len: %d, pair len: %d" % (data_args[0], data_args[1]))
    # max_mem = _benchmark_evoformer_stack_origin(data_args, get_model, get_data)
    # for ratio in [0.5, 0.4, 0.3, 0.2, 0.1]:
    #     try:
    #         _benchmark_evoformer_stack_gm(data_args, max_mem * ratio, get_model, get_data)
    #     except RuntimeError as e:
    #         if e.args[0] == "Search failed. Try a larger memory threshold.":
    #             break
    #     except Exception as e:
    #         raise e
    _benchmark_evoformer_stack_gm(data_args, None, get_model, get_data)


# FASTNN
# import torch

# from fastfold.model.fastnn import EvoformerStack, ExtraMSAStack
# from fastfold.model.fastnn.embedders import TemplateEmbedder
# from fastfold.model.fastnn.embedders_multimer import TemplateEmbedderMultimer
# from fastfold.model.fastnn.ops import RecyclingEmbedder, InputEmbedder
# from fastfold.model.nn.triangular_multiplicative_update import is_fused_triangle_multiplication


# def copy_layernorm(model_fast, model_ori):
#     model_fast.weight.copy_(model_ori.weight)
#     model_fast.bias.copy_(model_ori.bias)


# def copy_linear(model_fast, model_ori):
#     model_fast.weight.copy_(model_ori.weight)
#     if model_fast.use_bias:
#         model_fast.bias.copy_(model_ori.bias)


# def copy_native_linear(model_fast, model_ori):
#     model_fast.weight.copy_(model_ori.weight)
#     try:
#         model_fast.bias.copy_(model_ori.bias)
#     except:
#         pass


# def copy_kv_linear(model_fast, ori_k, ori_v):
#     model_fast.weight.copy_(torch.cat((ori_k.weight, ori_v.weight), dim=0))


# def copy_qkv_linear(model_fast, ori_q, ori_k, ori_v):
#     model_fast.weight.copy_(torch.cat((ori_q.weight, ori_k.weight, ori_v.weight), dim=0))


# def copy_attention(model_fast, model_ori):
#     copy_qkv_linear(model_fast.to_qkv, model_ori.linear_q, model_ori.linear_k, model_ori.linear_v)
#     copy_linear(model_fast.gating_linear, model_ori.linear_g)
#     copy_linear(model_fast.o_linear, model_ori.linear_o)

#     try:
#         model_fast.gating_bias.copy_(model_ori.linear_g.bias)
#     except:
#         print("no gating_bias need copy")


# def copy_left_right(model_fast, ori_left, ori_right):
#     model_fast.weight.copy_(torch.cat((ori_left.weight, ori_right.weight), dim=0))
#     model_fast.bias.copy_(torch.cat((ori_left.bias, ori_right.bias), dim=0))


# def copy_transition(model_fast, model_ori):
#     copy_layernorm(model_fast.norm, model_ori.layer_norm)
#     copy_linear(model_fast.linear1, model_ori.linear_1)
#     copy_linear(model_fast.linear2, model_ori.linear_2)


# def copy_triangle(model_fast, model_ori):
#     copy_layernorm(model_fast.layernorm1, model_ori.layer_norm_in)
#     copy_layernorm(model_fast.layernorm2, model_ori.layer_norm_out)

#     copy_linear(model_fast.output_projection, model_ori.linear_z)
#     model_fast.output_bias.copy_(model_ori.linear_z.bias)

#     if is_fused_triangle_multiplication():
#         copy_linear(model_fast.output_gate, model_ori.linear_gate)
#         copy_linear(model_fast.left_right_projection, model_ori.linear_p)
#         copy_linear(model_fast.left_right_gate, model_ori.linear_g)
#     else:
#         copy_linear(model_fast.output_gate, model_ori.linear_g)
#         copy_left_right(model_fast.left_right_projection, model_ori.linear_a_p, model_ori.linear_b_p)
#         copy_left_right(model_fast.left_right_gate, model_ori.linear_a_g, model_ori.linear_b_g)


# def copy_triangle_att(model_fast, model_ori):
#     copy_layernorm(model_fast.layernorm1, model_ori.layer_norm)
#     copy_linear(model_fast.linear_b, model_ori.linear)
#     copy_attention(model_fast.attention, model_ori.mha)

#     model_fast.out_bias.copy_(model_ori.mha.linear_o.bias)


# def copy_native_att(model_fast, model_ori):
#     copy_native_linear(model_fast.linear_q, model_ori.linear_q)
#     copy_native_linear(model_fast.linear_k, model_ori.linear_k)
#     copy_native_linear(model_fast.linear_v, model_ori.linear_v)
#     copy_native_linear(model_fast.linear_o, model_ori.linear_o)
#     if model_ori.gating:
#          copy_native_linear(model_fast.linear_g, model_ori.linear_g)


# def copy_evoformer_para(block_fast, block_ori):
#     # msa_stack
#     # MSARowAttentionWithPairBias
#     copy_layernorm(block_fast.msa.MSARowAttentionWithPairBias.layernormM,
#                    block_ori.msa_att_row.layer_norm_m)
#     copy_layernorm(block_fast.msa.MSARowAttentionWithPairBias.layernormZ,
#                    block_ori.msa_att_row.layer_norm_z)

#     copy_attention(block_fast.msa.MSARowAttentionWithPairBias.attention,
#                    block_ori.msa_att_row.mha)

#     block_fast.msa.MSARowAttentionWithPairBias.linear_b_weights.copy_(
#         block_ori.msa_att_row.linear_z.weight)

#     block_fast.msa.MSARowAttentionWithPairBias.out_bias.copy_(
#         block_ori.msa_att_row.mha.linear_o.bias)

#     # MSAColumnAttention
#     copy_layernorm(block_fast.msa.MSAColumnAttention.layernormM,
#                    block_ori.msa_att_col._msa_att.layer_norm_m)

#     copy_attention(block_fast.msa.MSAColumnAttention.attention,
#                    block_ori.msa_att_col._msa_att.mha)

#     # MSATransition
#     copy_transition(block_fast.msa.MSATransition, block_ori.core.msa_transition)

#     # communication
#     copy_layernorm(block_fast.communication.layernormM,
#                    block_ori.core.outer_product_mean.layer_norm)
#     copy_linear(block_fast.communication.linear_a, block_ori.core.outer_product_mean.linear_1)
#     copy_linear(block_fast.communication.linear_b, block_ori.core.outer_product_mean.linear_2)
#     copy_linear(block_fast.communication.o_linear, block_ori.core.outer_product_mean.linear_out)

#     # pair_stack
#     # TriangleMultiplicationOutgoing
#     copy_triangle(block_fast.pair.TriangleMultiplicationOutgoing, block_ori.core.tri_mul_out)
#     # TriangleMultiplicationIncoming
#     copy_triangle(block_fast.pair.TriangleMultiplicationIncoming, block_ori.core.tri_mul_in)

#     # TriangleAttentionStartingNode
#     copy_triangle_att(block_fast.pair.TriangleAttentionStartingNode,
#                       block_ori.core.tri_att_start)
#     copy_triangle_att(block_fast.pair.TriangleAttentionEndingNode, block_ori.core.tri_att_end)

#     copy_transition(block_fast.pair.PairTransition, block_ori.core.pair_transition)


# def copy_global_attention(model_fast, model_ori):
#     copy_linear(model_fast.to_q, model_ori.linear_q)
#     copy_kv_linear(model_fast.to_kv, model_ori.linear_k, model_ori.linear_v)
#     copy_linear(model_fast.gating_linear, model_ori.linear_g)
#     copy_linear(model_fast.o_linear, model_ori.linear_o)

#     try:
#         model_fast.gating_bias.copy_(model_ori.linear_g.bias)
#     except:
#         print("no gating_bias need copy")


# def copy_extra_msa_para(block_fast, block_ori):
#     # msa_stack
#     # MSARowAttentionWithPairBias
#     copy_layernorm(
#         block_fast.msa_stack.MSARowAttentionWithPairBias.layernormM,
#         block_ori.msa_att_row.layer_norm_m,
#     )
#     copy_layernorm(
#         block_fast.msa_stack.MSARowAttentionWithPairBias.layernormZ,
#         block_ori.msa_att_row.layer_norm_z,
#     )

#     copy_attention(
#         block_fast.msa_stack.MSARowAttentionWithPairBias.attention,
#         block_ori.msa_att_row.mha,
#     )

#     block_fast.msa_stack.MSARowAttentionWithPairBias.linear_b_weights.copy_(
#         block_ori.msa_att_row.linear_z.weight
#     )

#     block_fast.msa_stack.MSARowAttentionWithPairBias.out_bias.copy_(
#         block_ori.msa_att_row.mha.linear_o.bias
#     )

#     # MSAColumnAttention
#     copy_layernorm(
#         block_fast.msa_stack.MSAColumnAttention.layernormM,
#         block_ori.msa_att_col.layer_norm_m,
#     )

#     copy_global_attention(
#         block_fast.msa_stack.MSAColumnAttention.global_attention,
#         block_ori.msa_att_col.global_attention,
#     )

#     # MSATransition
#     copy_transition(block_fast.msa_stack.MSATransition, block_ori.core.msa_transition)

#     # communication
#     comm_model = (
#         block_ori.core.outer_product_mean# if not block_ori.is_multimer else block_ori.outer_product_mean
#     )
#     copy_layernorm(block_fast.communication.layernormM, comm_model.layer_norm)
#     copy_linear(block_fast.communication.linear_a, comm_model.linear_1)
#     copy_linear(block_fast.communication.linear_b, comm_model.linear_2)
#     copy_linear(block_fast.communication.o_linear, comm_model.linear_out)

#     # pair_stack
#     # TriangleMultiplicationOutgoing
#     copy_triangle(
#         block_fast.pair_stack.TriangleMultiplicationOutgoing, block_ori.core.tri_mul_out
#     )
#     # TriangleMultiplicationIncoming
#     copy_triangle(
#         block_fast.pair_stack.TriangleMultiplicationIncoming, block_ori.core.tri_mul_in
#     )

#     # TriangleAttentionStartingNode
#     copy_triangle_att(
#         block_fast.pair_stack.TriangleAttentionStartingNode,
#         block_ori.core.tri_att_start,
#     )
#     copy_triangle_att(
#         block_fast.pair_stack.TriangleAttentionEndingNode, block_ori.core.tri_att_end
#     )

#     copy_transition(
#         block_fast.pair_stack.PairTransition, block_ori.core.pair_transition
#     )


# def copy_template_pair_stack_para(block_fast, block_ori):
#     # TriangleMultiplicationOutgoing
#     copy_triangle(block_fast.TriangleMultiplicationOutgoing, block_ori.tri_mul_out)
#     # TriangleMultiplicationIncoming
#     copy_triangle(block_fast.TriangleMultiplicationIncoming, block_ori.tri_mul_in)

#     # TriangleAttentionStartingNode
#     copy_triangle_att(block_fast.TriangleAttentionStartingNode, block_ori.tri_att_start)
#     copy_triangle_att(block_fast.TriangleAttentionEndingNode, block_ori.tri_att_end)

#     copy_transition(block_fast.PairTransition, block_ori.pair_transition)


# def copy_template_pair_block_para(fast_module, target_module):
#     with torch.no_grad():
#         for ori_block, fast_block in zip(target_module.blocks, fast_module.blocks):
#             copy_template_pair_stack_para(fast_block, ori_block)
#             if ori_block.training == False:
#                 fast_block.eval()


# def copy_template_para(block_fast, block_ori):
#     # TemplateAngleEmbedder
#     copy_linear(block_fast.template_angle_embedder.linear_1,
#                 block_ori.template_angle_embedder.linear_1)
#     copy_linear(block_fast.template_angle_embedder.linear_2,
#                 block_ori.template_angle_embedder.linear_2)

#     # TemplatePairEmbedder
#     copy_linear(block_fast.template_pair_embedder.linear,
#                 block_ori.template_pair_embedder.linear)

#     # TemplatePairStack
#     copy_template_pair_block_para(block_fast.template_pair_stack,
#                                   block_ori.template_pair_stack)
#     copy_layernorm(block_fast.template_pair_stack.layer_norm,
#                    block_ori.template_pair_stack.layer_norm)

#     # TemplatePointwiseAttention
#     copy_native_att(block_fast.template_pointwise_att.mha,
#                     block_ori.template_pointwise_att.mha)


# def copy_template_multimer_para(block_fast, block_ori):
#     # TemplatePairEmbedderMultimer
#     copy_linear(block_fast.template_pair_embedder.dgram_linear,
#                 block_ori.template_pair_embedder.dgram_linear)
#     copy_linear(block_fast.template_pair_embedder.aatype_linear_1,
#                 block_ori.template_pair_embedder.aatype_linear_1)
#     copy_linear(block_fast.template_pair_embedder.aatype_linear_2,
#                 block_ori.template_pair_embedder.aatype_linear_2)
#     copy_layernorm(block_fast.template_pair_embedder.query_embedding_layer_norm,
#                    block_ori.template_pair_embedder.query_embedding_layer_norm)
#     copy_linear(block_fast.template_pair_embedder.query_embedding_linear,
#                 block_ori.template_pair_embedder.query_embedding_linear)
#     copy_linear(block_fast.template_pair_embedder.pseudo_beta_mask_linear,
#                 block_ori.template_pair_embedder.pseudo_beta_mask_linear)
#     copy_linear(block_fast.template_pair_embedder.x_linear,
#                 block_ori.template_pair_embedder.x_linear)
#     copy_linear(block_fast.template_pair_embedder.y_linear,
#                 block_ori.template_pair_embedder.y_linear)
#     copy_linear(block_fast.template_pair_embedder.z_linear,
#                 block_ori.template_pair_embedder.z_linear)
#     copy_linear(block_fast.template_pair_embedder.backbone_mask_linear,
#                 block_ori.template_pair_embedder.backbone_mask_linear)

#     # TemplateSingleEmbedderMultimer
#     copy_linear(block_fast.template_single_embedder.template_single_embedder,
#                 block_ori.template_single_embedder.template_single_embedder)
#     copy_linear(block_fast.template_single_embedder.template_projector,
#                 block_ori.template_single_embedder.template_projector)

#     # TemplatePairStack
#     copy_template_pair_block_para(block_fast.template_pair_stack,
#                                   block_ori.template_pair_stack)
#     copy_layernorm(block_fast.template_pair_stack.layer_norm,
#                    block_ori.template_pair_stack.layer_norm)

#     # linear_t
#     copy_linear(block_fast.linear_t, block_ori.linear_t)


# def inject_evoformer(model):
#     with torch.no_grad():
#         target_module = model.evoformer
#         fast_module = EvoformerStack(
#             c_m=target_module.blocks[0].msa_att_row.c_in,
#             c_z=target_module.blocks[0].msa_att_row.c_z,
#             c_s=target_module.linear.out_features,
#             no_blocks=len(target_module.blocks),
#             blocks_per_ckpt=target_module.blocks_per_ckpt,
#             clear_cache_between_blocks=target_module.clear_cache_between_blocks,
#             is_multimer=target_module.blocks[0].is_multimer,
#         )
#         for target_block, fast_block in zip(target_module.blocks, fast_module.blocks):
#             copy_evoformer_para(fast_block, target_block)
#             if target_block.training == False:
#                 fast_block.eval()
#         copy_linear(fast_module.linear, target_module.linear)
#         model.evoformer = fast_module
#         # print("\nNew model.evoformer\n")
#         # print(model.evoformer)

# # TODO: new addition
# def inject_evoformer1(model):
#     with torch.no_grad():
#         target_module = model.evoformer
#         fast_module = EvoformerStack(
#             c_m=target_module.blocks[0].msa_att_row.c_in,
#             c_z=target_module.blocks[0].msa_att_row.c_z,
#             c_s=target_module.linear.out_features,
#             no_blocks=len(target_module.blocks),
#             blocks_per_ckpt=target_module.blocks_per_ckpt,
#             clear_cache_between_blocks=target_module.clear_cache_between_blocks,
#             is_multimer=target_module.blocks[0].is_multimer,
#         )
#         for target_block, fast_block in zip(target_module.blocks, fast_module.blocks):
#             copy_evoformer_para(fast_block, target_block)
#             if target_block.training == False:
#                 fast_block.eval()
#         copy_linear(fast_module.linear, target_module.linear)



# def inject_extramsa(model):
#     with torch.no_grad():
#         target_module = model.extra_msa_stack
#         fast_module = ExtraMSAStack(
#             c_m=target_module.blocks[0].msa_att_row.c_in,
#             c_z=target_module.blocks[0].msa_att_row.c_z,
#             no_blocks=len(target_module.blocks),
#             clear_cache_between_blocks=target_module.clear_cache_between_blocks,
#             is_multimer=target_module.blocks[0].is_multimer,
#         )
#         for target_block, fast_block in zip(target_module.blocks, fast_module.blocks):
#             copy_extra_msa_para(fast_block, target_block)
#             if target_block.training == False:
#                 fast_block.eval()
#         model.extra_msa_stack = fast_module


# def inject_template(model):
#     with torch.no_grad():
#         if model.evoformer.blocks[0].is_multimer:
#             target_module = model.template_embedder
#             fast_module = TemplateEmbedderMultimer(config=model.template_embedder.config)
#             copy_template_multimer_para(fast_module, target_module)
#             if target_module.training == False:
#                 fast_module.eval()
#             model.template_embedder = fast_module
#         else:
#             target_module = model.template_embedder
#             fast_module = TemplateEmbedder(config=model.template_embedder.config)
#             copy_template_para(fast_module, target_module)
#             if target_module.training == False:
#                 fast_module.eval()
#             model.template_embedder = fast_module


# def inject_embedder(model):
#     if model.evoformer.blocks[0].is_multimer:
#         return

#     # recycle embedder
#     with torch.no_grad():
#         target_module = model.recycling_embedder
#         fast_module = RecyclingEmbedder(
#             c_m=target_module.c_m,
#             c_z=target_module.c_z,
#             min_bin=target_module.min_bin,
#             max_bin=target_module.max_bin,
#             no_bins=target_module.no_bins,
#             inf=target_module.inf
#         )
#         copy_native_linear(fast_module.linear, target_module.linear)
#         copy_layernorm(fast_module.layer_norm_m, target_module.layer_norm_m)
#         copy_layernorm(fast_module.layer_norm_z, target_module.layer_norm_z)
#         if target_module.training == False:
#             fast_module.eval()
#         model.recycling_embedder = fast_module

#     # input embedder
#     with torch.no_grad():
#         target_module = model.input_embedder
#         fast_module = InputEmbedder(
#             tf_dim=target_module.tf_dim,
#             msa_dim=target_module.msa_dim,
#             c_z=target_module.c_z,
#             c_m=target_module.c_m,
#             relpos_k=target_module.relpos_k,
#         )
#         copy_linear(fast_module.linear_tf_z_i, target_module.linear_tf_z_i)
#         copy_linear(fast_module.linear_tf_z_j, target_module.linear_tf_z_j)
#         copy_linear(fast_module.linear_tf_m, target_module.linear_tf_m)
#         copy_linear(fast_module.linear_msa_m, target_module.linear_msa_m)
#         copy_linear(fast_module.linear_relpos, target_module.linear_relpos)
#         if target_module.training == False:
#             fast_module.eval()
#         model.input_embedder = fast_module


# def inject_fastnn(model):
#     print("\nInjected fastnn\n")
#     inject_evoformer(model)
#     inject_extramsa(model)
#     inject_template(model)
#     inject_embedder(model)
#     return model
# FASTNN


if __name__ == "__main__":
    # launch colossalai
    colossalai.launch(
        config={},
        rank=0,
        world_size=1,
        host="localhost",
        port=free_port(),
        backend="nccl",
    )
    benchmark_evoformer_stack((128, 128))
    # benchmark_evoformer_stack((256, 256))
    # benchmark_evoformer_stack((256, 512))
    # # TODO: OOM
    # benchmark_evoformer_stack((256, 1024))
    # benchmark_evoformer_stack((256, 1280))
