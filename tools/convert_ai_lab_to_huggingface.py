#!/usr/bin/env python
# coding:utf8



import argparse
import collections
import json
import os
import torch

parser = argparse.ArgumentParser()

parser.add_argument("--ailab_model_path", default=None, type=str, required=True,
                    help="The Tencent AI Lab model path.")
parser.add_argument("--ailab_vocab_path", default=None, type=str, required=True,
                    help="The Tencet AI Lab vocab file path.")
parser.add_argument("--huggingface_dump_path", default=None, type=str, required=True,
                    help="The output path, it is a dir!")
args = parser.parse_args()


def gen_huggingface_bert_config(ailab_bert_config, converted_ailab_bert_params):
    huggingface_bert_config = dict()
    huggingface_bert_config["attention_probs_dropout_prob"] = ailab_bert_config.dropout
    huggingface_bert_config["diectionality"] = "bidi"
    huggingface_bert_config["hidden_act"] = "gelu"
    huggingface_bert_config["hidden_dropout_prob"] = ailab_bert_config.dropout
    huggingface_bert_config["hidden_size"] = ailab_bert_config.embed_dim
    huggingface_bert_config["initializer_range"] = 0.02
    huggingface_bert_config["intermediate_size"] = ailab_bert_config.ff_embed_dim
    huggingface_bert_config["max_position_embeddings"] = \
        converted_ailab_bert_params["embeddings.position_embeddings.weight"].size(0)
    huggingface_bert_config["num_attention_heads"] = ailab_bert_config.num_heads
    huggingface_bert_config["num_hidden_layers"] = ailab_bert_config.layers
    huggingface_bert_config["pooler_fc_size"] = ailab_bert_config.embed_dim
    huggingface_bert_config["pooler_type"] = "first_token_transform"
    huggingface_bert_config["type_vocab_size"] = \
        converted_ailab_bert_params["embeddings.token_type_embeddings.weight"].size(0)
    huggingface_bert_config["vocab_size"] = \
        converted_ailab_bert_params["embeddings.word_embeddings.weight"].size(0)
    return huggingface_bert_config


def gen_huggingface_bert_model(ailab_bert_params):
    huggingface_bert_params = collections.OrderedDict()
    for k, v in ailab_bert_params.items():
        if k == 'tok_embed.weight':
            huggingface_bert_params['embeddings.word_embeddings.weight'] = v
        if k == 'pos_embed.weights.weight':
            huggingface_bert_params['embeddings.position_embeddings.weight'] = v
        if k == 'seg_embed.weight':
            huggingface_bert_params['embeddings.token_type_embeddings.weight'] = v
        if k == 'emb_layer_norm.weight':
            huggingface_bert_params['embeddings.LayerNorm.gamma'] = v
        if k == 'emb_layer_norm.bias':
            huggingface_bert_params['embeddings.LayerNorm.beta'] = v
        if k.endswith('in_proj_weight'):
            num_layer = k.split('.')[1]
            new_name = 'encoder.layer.' + num_layer + '.attention.self.'
            huggingface_bert_params[new_name + 'query.weight'], \
            huggingface_bert_params[new_name + 'key.weight'], \
            huggingface_bert_params[new_name + 'value.weight'] = v.chunk(3, dim=0)
        if k.endswith('in_proj_bias'):
            num_layer = k.split('.')[1]
            new_name = 'encoder.layer.' + num_layer + '.attention.self.'
            huggingface_bert_params[new_name + 'query.bias'], \
            huggingface_bert_params[new_name + 'key.bias'], \
            huggingface_bert_params[new_name + 'value.bias'] = v.chunk(3, dim=0)
        if k.endswith('out_proj.weight'):
            num_layer = k.split('.')[1]
            new_name = 'encoder.layer.' + num_layer + '.attention.output.dense.weight'
            huggingface_bert_params[new_name] = v
        if k.endswith('out_proj.bias'):
            num_layer = k.split('.')[1]
            new_name = 'encoder.layer.' + num_layer + '.attention.output.dense.bias'
            huggingface_bert_params[new_name] = v
        if k.endswith('attn_layer_norm.weight'):
            num_layer = k.split('.')[1]
            new_name = 'encoder.layer.' + num_layer + '.attention.output.LayerNorm.gamma'
            huggingface_bert_params[new_name] = v
        if k.endswith('attn_layer_norm.bias'):
            num_layer = k.split('.')[1]
            new_name = 'encoder.layer.' + num_layer + '.attention.output.LayerNorm.beta'
            huggingface_bert_params[new_name] = v
        if k.endswith('fc1.weight'):
            num_layer = k.split('.')[1]
            new_name = 'encoder.layer.' + num_layer + '.intermediate.dense.weight'
            huggingface_bert_params[new_name] = v
        if k.endswith('fc1.bias'):
            num_layer = k.split('.')[1]
            new_name = 'encoder.layer.' + num_layer + '.intermediate.dense.bias'
            huggingface_bert_params[new_name] = v
        if k.endswith('fc2.weight'):
            num_layer = k.split('.')[1]
            new_name = 'encoder.layer.' + num_layer + '.output.dense.weight'
            huggingface_bert_params[new_name] = v
        if k.endswith('fc2.bias'):
            num_layer = k.split('.')[1]
            new_name = 'encoder.layer.' + num_layer + '.output.dense.bias'
            huggingface_bert_params[new_name] = v
        if k.endswith('ff_layer_norm.weight'):
            num_layer = k.split('.')[1]
            new_name = 'encoder.layer.' + num_layer + '.output.LayerNorm.gamma'
            huggingface_bert_params[new_name] = v
        if k.endswith('ff_layer_norm.bias'):
            num_layer = k.split('.')[1]
            new_name = 'encoder.layer.' + num_layer + '.output.LayerNorm.beta'
            huggingface_bert_params[new_name] = v
        if k.startswith('one_more_nxt_snt'):
            _, suffix = k.split('.')
            huggingface_bert_params['pooler.dense.' + suffix] = v
    return huggingface_bert_params


def gen_huggingface_vocab(ailab_bert_vocab_file, vocab_size):
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    with open(ailab_bert_vocab_file, "r") as f:
        for line in f:
            data = line.strip().split("\t")
            if len(vocab) < vocab_size:
                vocab.append(data[0])
            else:
                break
    return vocab


def convert():
    ailab_bert_model = torch.load(args.ailab_model_path, map_location="cpu")
    huggingface_model = gen_huggingface_bert_model(ailab_bert_model["model"])
    huggingface_config = gen_huggingface_bert_config(ailab_bert_model['args'],
                                                     huggingface_model)
    huggingface_vocab = gen_huggingface_vocab(args.ailab_vocab_path,
                                              huggingface_config["vocab_size"])
    if not os.path.exists(args.huggingface_dump_path):
        os.mkdir(args.huggingface_dump_path)
    torch.save(huggingface_model,
               os.path.join(args.huggingface_dump_path, "ailab_model.bin"))
    with open(os.path.join(args.huggingface_dump_path, "ailab_model_config.json"), "w") as f:
        json.dump(huggingface_config, f)
    with open(os.path.join(args.huggingface_dump_path, "ailab_model.vocab"), "w") as f:
        for token in huggingface_vocab:
            f.writelines(token + "\n")


if __name__ == "__main__":
    convert()
