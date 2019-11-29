#!/usr/bin/env python
# coding:utf8

import argparse
import numpy as np
import json
import logging
import torch
import torch.utils.data as data

from models.classification import BertForMultiLabelClassification
from huggingface.modeling import BertConfig
from datautils.text_classification_dataset import MultiLabelClassificationDataset
from datautils.text_classification_dataset import MultiLabelClassificationCollate
from tqdm import tqdm
from utils import load_json_config, init_bert_adam_optimizer, load_saved_model, save_model

logging.basicConfig(level=logging.INFO,format="%(asctime)s - %(levelname)s - %(message)s")
def model_predict(master_gpu_id, model, dataset, config, eval_batch_size=1, use_cuda=False, num_workers=1, output_file='predict_output.txt'):
    model.eval()
    dataset.is_training = False
    eval_dataloader = data.DataLoader(dataset=dataset,
                                      collate_fn=MultiLabelClassificationCollate(dataset),
                                      pin_memory=use_cuda,
                                      batch_size=eval_batch_size,
                                      num_workers=num_workers,
                                      shuffle=False)
    correct_sum = 0
    num_sample = eval_dataloader.dataset.testing_len
    true_labels = []
    logging.info("Predicting Model...")
    logging.info("output_file=%s" % output_file)

    ofile = open(output_file, 'w')
    for batch in tqdm(eval_dataloader, mininterval=5, unit="batch", ncols=100, desc="Predicting process: "):
        tokens = batch["tokens"].cuda(master_gpu_id) if use_cuda else batch["tokens"]
        segment_ids = batch["segment_ids"].cuda(master_gpu_id) if use_cuda else batch["segment_ids"]
        attn_masks = batch["attn_masks"].cuda(master_gpu_id) if use_cuda else batch["attn_masks"]
        labels = batch["labels"].cuda(master_gpu_id) if use_cuda else batch["labels"]
        labels_idx = batch["labels_idx"]
        contents = batch["contents"]
        vids = batch["vids"]
        with torch.no_grad():
            logit = model(tokens, segment_ids, attn_masks)
        top_probs, top_index = logit.topk(4)
        logit = logit.tolist()
        for vid,content, lg, lbidx, topidx, topprob in zip(vids,contents, logit, labels_idx, top_index.tolist(), top_probs.tolist()):
            lbidx_labels = ["%s" %(dataset.idx2label[idx]) for idx in lbidx]
            topidx_labels = ["%s" %(dataset.idx2label[idx]) for idx in topidx]
            print(vid, content, '\t', "!;".join(sorted(lbidx_labels)), '\t', "!;".join(sorted(topidx_labels)), file=ofile)
    ofile.close()


def get_label_idx(labels):
    res = []
    for batch, label in enumerate(labels):
        b = []
        for idx, i in enumerate(label):
            if i == 1:
                b.append(idx)
        res.append(b)
    return res


def eval_model(master_gpu_id, model, dataset, label_size, eval_batch_size=1,use_cuda=False, num_workers=1):
    """Evaluate the performance for text classifier.
    Args:
        master_gpu_id: id of master gpu
        model: the fine-tuned BERT model
        dataset: the evaluating dataset
        eval_batch_size: batch size for evaluating
        use_cuda: whether to use cuda
        num_workers: use how many processes to load dataset.
    """
    model.eval()
    dataset.is_training = False
    eval_dataloader = data.DataLoader(dataset=dataset,
                                      collate_fn=MultiLabelClassificationCollate(dataset),
                                      pin_memory=use_cuda,
                                      batch_size=eval_batch_size,
                                      num_workers=num_workers,
                                      shuffle=False)
    correct_sum = 0
    num_sample = eval_dataloader.dataset.testing_len
    predicted_probs = []
    true_labels = []
    logging.info("Evaluating Model...")

    prf_list = []

    for batch in tqdm(eval_dataloader, mininterval=5, unit="batch", ncols=100, desc="Evaluating process: "):
        tokens = batch["tokens"].cuda(master_gpu_id) if use_cuda else batch["tokens"]
        segment_ids = batch["segment_ids"].cuda(master_gpu_id) if use_cuda else batch["segment_ids"]
        attn_masks = batch["attn_masks"].cuda(master_gpu_id) if use_cuda else batch["attn_masks"]
        labels = batch["labels"].cuda(master_gpu_id) if use_cuda else batch["labels"]
        labels_idx = batch["labels_idx"]
        with torch.no_grad():
            logit = model(tokens, segment_ids, attn_masks)
        _, top_index = logit.topk(4)
        pred_labels = top_index.tolist()

        def get_prf(truth_label, pred_label):
            iou, precision, recall, f1_score = 0, 0, 0, 0
            truth = set(truth_label)
            pred = set(pred_label)
            i_count = len([x for x in pred if x in truth])
            u_count = len(set(pred).union(set(truth)))
            if i_count > 0:
                iou = i_count / float(u_count)
            if len(pred) > 0:
                precision = i_count / float(len(pred))
            if len(truth) > 0:
                recall = i_count / float(len(truth))
            if precision + recall > 0:
                f1_score = precision * recall * 2 / (precision + recall)
            return [iou, precision, recall, f1_score]


        for truth_label, pred_label in zip(labels_idx, pred_labels):

            prf = get_prf(truth_label, pred_label)
            prf_list.append(prf)
            iou_res = np.array(prf_list).mean(axis=0)
    logging.info('iou: %.4f, precision: %.4f, recall: %.4f, f1_score: %.4f' % (iou_res[0],iou_res[1],iou_res[2],iou_res[3]))

def train_epoch(master_gpu_id, model, optimizer, dataloader, gradient_accumulation_steps, use_cuda):
    """Training process for each epoch.
    Args:
        master_gpu_id: id of master gpu
        model: model that need to be trained
        optimizer: the optimizer of the model
        dataloader: dataloader for TRAINING
        gradient_accumulation_steps: how many steps to accumulate gradient
        use_cuda: whether to use cuda
    Returns:
        the average losses of each batch
    """
    model.train()
    dataloader.dataset.is_training = True
    total_loss = 0.0
    total_pos_loss = 0.0
    total_neg_loss = 0.0
    correct_sum = 0
    total_sum = 0
    num_batch = dataloader.__len__()
    num_sample = dataloader.dataset.__len__()
    for step, batch in enumerate(dataloader):

        tokens = batch["tokens"].cuda(master_gpu_id) if use_cuda else batch["tokens"]
        segment_ids = batch["segment_ids"].cuda(master_gpu_id) if use_cuda else batch["segment_ids"]
        attn_masks = batch["attn_masks"].cuda(master_gpu_id) if use_cuda else batch["attn_masks"]
        labels = batch["labels"].cuda(master_gpu_id) if use_cuda else batch["labels"]
        loss, pos_loss, neg_loss, logit = model(tokens, segment_ids, attn_masks, labels)
        loss = loss.mean()
        if gradient_accumulation_steps > 1:
            loss /= gradient_accumulation_steps
        loss.backward()
        optimizer.step()
        model.zero_grad()
        loss_val = loss.item()
        pos_loss_val = pos_loss.mean().item()
        neg_loss_val = neg_loss.mean().item()
        total_loss += loss_val
        total_pos_loss += pos_loss_val
        total_neg_loss += neg_loss_val

        if (step + 1) % 100 == 0 or step == 0:
            logging.info("loss: %.5f, pos loss: %.5f, neg loss: %.5f" %(loss_val, pos_loss_val, neg_loss_val))
    logging.info("loss: %.5f " %(total_loss / num_batch))
    return total_loss / num_batch

def train_model(experiment_name, master_gpu_id, model, optimizer, epochs, dataset, label_size,
                batch_size=1, eval_batch_size=1, gradient_accumulation_steps=1,
                use_cuda=False, num_workers=1):
    logging.info("Start Training".center(60, "="))
    training_dataloader = data.DataLoader(dataset=dataset,
                                          collate_fn=MultiLabelClassificationCollate(dataset),
                                          pin_memory=use_cuda,
                                          batch_size=batch_size,
                                          num_workers=num_workers,
                                          shuffle=True)
    for epoch in range(1, epochs + 1):
        logging.info("Training Epoch: " + str(epoch))
        avg_loss = train_epoch(master_gpu_id, model, optimizer, training_dataloader,gradient_accumulation_steps, use_cuda)
        logging.info("Average Loss: " + format(avg_loss, "0.4f"))

        eval_model(master_gpu_id, model, dataset, label_size, eval_batch_size, use_cuda, num_workers)
        save_model(experiment_name, model, epoch)


def main(args):
    logging.info("Loading HyperParameters".center(60, "="))
    config = load_json_config(args.config_file)
    logging.info(json.dumps(config, indent=2, sort_keys=True))
    logging.info("Load HyperParameters Done".center(60, "="))

    logging.info("Loading Dataset".center(60, "="))
    dataset = MultiLabelClassificationDataset(vocab_file=config.get("vocab_file"),
                                              label_file=config.get("label_file"),
                                              label_weight_file=config.get("label_weight_file"),
                                              max_seq_len=config.get("max_seq_len"),
                                              training_path=config.get("training_path"),
                                              testing_path=config.get("testing_path"))
    
    logging.info("Total training line: " + str(dataset.training_len) +  ", total testing line: " + str(dataset.testing_len))
    label_size = len(dataset.label2idx)
    logging.info('label size: %d' % label_size)
    logging.info("Load Dataset Done".center(60, "="))
    label_weight = dataset.label_weight.to('cuda') if config.get("use_cuda") else dataset.label_weight

    logging.info("Initializing SequenceClassification Model".center(60, "="))
    if config.get("pretrained_model_path"):
        model = BertForMultiLabelClassification.load_pretrained_bert_model(
            bert_config_path=config.get("bert_config_path"),
            pretrained_model_path=config.get("pretrained_model_path"),
            num_labels=len(dataset.label2idx), label_weight=label_weight)
    else:
        model = BertForMultiLabelClassification(BertConfig.from_json_file(config.get("bert_config_path")),len(dataset.label2idx), label_weight=label_weight)
    if config.get("num_tuning_layers") is not None:
        model.bert.encoder.layer = torch.nn.ModuleList(
            model.bert.encoder.layer[:config.get("num_tuning_layers")])
    logging.info(model)
    logging.info("Initialize SequenceClassification Model Done".center(60, "="))

    if args.saved_model:
        logging.info("Loading Saved Model".center(60, "="))
        logging.info("Load saved model from: " + args.saved_model)
        load_saved_model(model, args.saved_model)
        logging.info("Load Saved Model Done".center(60, "="))

    master_gpu_id = None
    if len(args.gpu_ids) == 1:
        master_gpu_id = int(args.gpu_ids)
        model = model.cuda(int(args.gpu_ids)) if config.get("use_cuda") else model
    else:
        gpu_ids = [int(each) for each in args.gpu_ids.split(",")]
        master_gpu_id = gpu_ids[0]
        model = model.cuda(gpu_ids[0])
        logging.info("Start multi-gpu dataparallel training/evaluating...")
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)

    if args.mode == "eval":
        if args.input_file:
            dataset = MultiLabelClassificationDataset(vocab_file=config.get("vocab_file"),
                                                label_file=config.get("label_file"),
                                                max_seq_len=config.get("max_seq_len"),
                                                label_weight_file=config.get("label_weight_file"),
                                                testing_path=args.input_file)
        eval_model(master_gpu_id, model, dataset, label_size, config.get("eval_batch_size"),config.get("use_cuda"), config.get("num_workers"))

    elif args.mode == "predict":
        if args.input_file:
            dataset = MultiLabelClassificationDataset(vocab_file=config.get("vocab_file"),
                                                label_file=config.get("label_file"),
                                                max_seq_len=config.get("max_seq_len"),
                                                label_weight_file=config.get("label_weight_file"),
                                                testing_path=args.input_file)

        model_predict(master_gpu_id, model, dataset, config, config.get("eval_batch_size"),config.get("use_cuda"), config.get("num_workers"), args.output_file)

    elif args.mode == "train":
        optimizer = init_bert_adam_optimizer(model, dataset.training_len,
                                             config.get("epochs"),
                                             config.get("batch_size"),
                                             config.get("gradient_accumulation_steps"),
                                             config.get("init_lr"),
                                             config.get("warmup_proportion"))
        train_model(config.get("experiment_name"), master_gpu_id, model,
                    optimizer, config.get("epochs"), dataset, label_size,
                    batch_size=config.get("batch_size"),
                    eval_batch_size=config.get("eval_batch_size"),
                    gradient_accumulation_steps=config.get("gradient_accumulation_steps"),
                    use_cuda=config.get("use_cuda"),
                    num_workers=config.get("num_workers"))
    else:
        raise RuntimeError("Mode not support: " + args.mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Textclassification training script arguments.")
    parser.add_argument("-c", "--config", dest="config_file", action="store", default='/data/ceph_11015/ssd/anhan/tagging_kid/bert/config/multi_label.json', help="The path of configuration json file.")
    parser.add_argument("-s", "--savedmodel", dest="saved_model",default='/data/ceph_11015/ssd/anhan/tagging_kid/bert/checkpoints/v1118/Epoch_18.bin', action="store", help="The path of trained checkpoint model.")
    parser.add_argument("-m", "--mode", dest="mode", action="store", default="train",help="Running mode, train or eval.")
    parser.add_argument("-g", "--gpu", dest="gpu_ids", action="store", default="0",help="Device ids of used gpus, split by ','")
    parser.add_argument("--input_file", default='/data/ceph_11015/ssd/anhan/tagging_kid/bert/toy_data/data_v1118/test')
    parser.add_argument("--output_file", default='/data/ceph_11015/ssd/anhan/tagging_kid/bert/output.txt')
    parsed_args = parser.parse_args()
    main(parsed_args)