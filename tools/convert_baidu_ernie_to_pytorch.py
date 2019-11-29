#!/usr/bin/env python
# coding:utf8

# Convert Baidu ERNIE pretrained model to huggingface model structure.


import argparse
import collections
import os
import shutil
import subprocess
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("--ernie_params_path", default=None, type=str, required=True,
                    help="The params path of baidu ernie pretrained model.")
parser.add_argument("--huggingface_dump_path", default=None, type=str, required=True,
                    help="The output path, it is a dir!")
args = parser.parse_args()


def get_lark():
    if not os.path.exists("./LARK"):
        subprocess.call("git clone https://github.com/PaddlePaddle/LARK.git", shell=True)


def gen_huggingface_bert_model(params_path):
    import paddle.fluid as fluid
    import sys
    sys.path.append("./LARK/ERNIE")
    from model.ernie import ErnieConfig
    from finetune.classifier import create_model
    from utils.init import init_pretraining_params

    ernie_config = ErnieConfig("./LARK/ERNIE/config/ernie_config.json")
    startup_prog = fluid.default_startup_program()
    test_prog = fluid.Program()

    args.max_seq_len = 512
    args.use_fp16 = False
    args.num_labels = 2
    args.loss_scaling = 1.0
    with fluid.program_guard(test_prog, startup_prog):
        with fluid.unique_name.guard():
            _, _ = create_model(
                args,
                pyreader_name="test",
                ernie_config=ernie_config)

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_prog)
    init_pretraining_params(exe, params_path, main_program=startup_prog)
    sc = fluid.global_scope()
    new_model = collections.OrderedDict()
    for each in startup_prog.list_vars():
        name = each.name
        if name == "test_reader":
            continue
        fc_w = sc.find_var(name).get_tensor()
        fc_w = np.array(fc_w, dtype=np.float32)
        if name == "word_embedding":
            new_model["embeddings.word_embeddings.weight"] = fc_w
        if name == "pos_embedding":
            new_model["embeddings.position_embeddings.weight"] = fc_w
        if name == "sent_embedding":
            new_model["embeddings.token_type_embeddings.weight"] = fc_w
        if name == "pre_encoder_layer_norm_scale":
            new_model["embeddings.LayerNorm.gamma"] = fc_w
        if name == "pre_encoder_layer_norm_bias":
            new_model["embeddings.LayerNorm.beta"] = fc_w
        if name.startswith("encoder_layer_"):
            splits = name.split(".")
            if len(splits) == 2:
                prefix, suffix = splits
            else:
                prefix = splits[0]
            prefixs = prefix.split("_")
            if prefixs[3] == "multi":
                new_suffix = ".weight" if suffix == "w_0" else ".bias"
                if new_suffix == ".weight":
                    fc_w = fc_w.transpose()
                if prefixs[6] == "output":
                    all_name = "encoder.layer." + prefixs[2] + \
                        ".attention.output.dense" + new_suffix
                else:
                    all_name = "encoder.layer." + prefixs[2] + \
                        ".attention.self." + prefixs[6] + new_suffix
            elif prefixs[3] == "post":
                new_suffix = ".gamma" if name.endswith("scale") else ".beta"
                if prefixs[4] == "att":
                    all_name = "encoder.layer." + prefixs[2] + \
                        ".attention.output.LayerNorm" + new_suffix
                elif prefixs[4] == "ffn":
                    all_name = "encoder.layer." + prefixs[2] + \
                        ".output.LayerNorm" + new_suffix
            elif prefixs[3] == "ffn":
                new_suffix = ".weight" if suffix == "w_0" else ".bias"
                if new_suffix == ".weight":
                    fc_w = fc_w.transpose()
                if prefixs[5] == "0":
                    all_name = "encoder.layer." + prefixs[2] + \
                        ".intermediate.dense" + new_suffix
                elif prefixs[5] == "1":
                    all_name = "encoder.layer." + prefixs[2] + \
                        ".output.dense" + new_suffix
            new_model[all_name] = fc_w
        if name == "pooled_fc.w_0":
            fc_w = fc_w.transpose()
            new_model["pooler.dense.weight"] = fc_w
        if name == "pooled_fc.b_0":
            new_model["pooler.dense.bias"] = fc_w
    return new_model


def save_huggingface_model(dump_path, huggingface_model):
    npy_path = os.path.join(dump_path, "ernie_for_npy")
    if os.path.exists(npy_path):
        shutil.rmtree(npy_path)
    os.mkdir(npy_path)
    for k, v in huggingface_model.items():
        np.save(os.path.join(npy_path, k), v)
    subprocess.call("python tools/convert_npy_to_pytorch.py " + dump_path + " " + npy_path,
                    shell=True)
    shutil.rmtree(npy_path)


def save_config(dump_path):
    subprocess.call("cp ./LARK/ERNIE/config/ernie_config.json " + dump_path, shell=True)


def save_vocab(dump_path):
    out_f = open(os.path.join(dump_path, "vocab_simple.txt"), "w")
    with open("./LARK/ERNIE/config/vocab.txt", "r") as f:
        for line in f:
            data = line.strip().split("\t")
            if len(data) != 2:
                continue
            out_f.writelines(data[0] + "\n")
    out_f.close()


def convert():
    get_lark()
    huggingface_model = gen_huggingface_bert_model(args.ernie_params_path)
    if not os.path.exists(args.huggingface_dump_path):
        os.mkdir(args.huggingface_dump_path)
    save_huggingface_model(args.huggingface_dump_path, huggingface_model)
    save_config(args.huggingface_dump_path)
    save_vocab(args.huggingface_dump_path)


if __name__ == "__main__":
    convert()
