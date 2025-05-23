import torch
import numpy as np
from tqdm import tqdm
from utils import get_output_index, macro, micro
# from sklearn import metrics
import time
import pickle

def train(args, model, optimizer, epoch, gpu, data_loader, scheduler):

    print("EPOCH %d" % epoch)

    losses = []
    device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.__dict__["no_cuda"] else "cpu"
    )
    model.train()

    # loader
    data_iter = iter(data_loader)
    num_iter = len(data_loader)
    for i in tqdm(range(num_iter)):

        inputs_id, segment_ids, masks, labels = next(data_iter)

        inputs_id, segment_ids, masks, labels = torch.LongTensor(inputs_id), torch.LongTensor(segment_ids),\
                                                torch.LongTensor(masks), torch.tensor(labels)


        inputs_id, segment_ids, masks, labels = inputs_id.to(device),segment_ids.to(device), masks.to(device), labels.to(device)

        # inputs_id, segment_ids, masks, labels = inputs_id.cuda(
            # gpu), segment_ids.cuda(gpu), masks.cuda(gpu), labels.cuda(gpu)

        # logits_cand,logit_context, loss = model(inputs_id, segment_ids, masks, labels)
        logits_cand,logits_context, loss = model(None,None,None,None,None,None,inputs_id, segment_ids, masks, labels,mode="train")
        # print(logits_cand==logits_context)
        # print("++++++",logits_cand.shape)
        # print("=========",loss.shape)
        # print(loss)
        loss=loss.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

    return losses

def test(args, model, fold, gpu, data_loader):

    gold_pred = []
    device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.__dict__["no_cuda"] else "cpu"
    )
    # loader
    data_iter = iter(data_loader)
    num_iter = len(data_loader)
    for i in tqdm(range(num_iter),mininterval=10):
        with torch.no_grad():

            inputs_id, segment_ids, masks, labels ,labels_coarse= next(data_iter)

            inputs_id, segment_ids, masks, labels ,labels_coarse= torch.LongTensor(
                inputs_id), torch.LongTensor(segment_ids), torch.LongTensor(masks), torch.tensor(labels),torch.LongTensor(labels_coarse)

            inputs_id, segment_ids, masks, labels,labels_coarse = inputs_id.to(device),segment_ids.to(device), masks.to(device), labels.to(device),labels_coarse.to(device)

            # logits_cand,logits_context, loss = model(None,None,None,None,None,None,inputs_id, segment_ids, masks, labels,mode="test")
            _,_,_,logits_cand, loss = model(None,None,None,None,None,None,inputs_id, segment_ids, masks, labels,labels_coarse,mode="test")

            labels = labels.detach().cpu().numpy()
            # output_index = get_output_index(logits_context, threshold=0.5)
            output_index_cand = get_output_index(logits_cand, threshold=0.5)
            y_labels = []
            for label in labels:
                y_labels.append([i for i in range(len(label)) if label[i] == 1])
            gold_pred += list(zip(y_labels, output_index_cand))
            # gold_pred += (list(zip(y_labels, output_index))+list(zip(y_labels, output_index_cand)))
            # print(y_labels)
            # print("================")
            # print(output_index)

    _, _, _, prec_macro, rec_macro, f1_macro = macro(gold_pred)
    _, _, _, prec_micro, rec_micro, f1_micro = micro(gold_pred)


    print(fold)
    print("[MACRO] all precision, all recall, all f-measure")
    print("%.4f, %.4f, %.4f" % (prec_macro, rec_macro, f1_macro))
    print("[MICRO] all precision, all recall, all f-measure")
    print("%.4f, %.4f, %.4f" % (prec_micro, rec_micro, f1_micro))
    print()


def generate(args, model, fold, gpu, data_loader, world):

    types = {}

    # loader
    data_iter = iter(data_loader)
    num_iter = len(data_loader)
    for i in tqdm(range(num_iter)):
        with torch.no_grad():

            inputs_id, segment_ids, masks, document_id = next(data_iter)

            inputs_id, segment_ids, masks = torch.LongTensor(
                inputs_id), torch.LongTensor(segment_ids), torch.LongTensor(masks)

            inputs_id, segment_ids, masks = inputs_id.cuda(
                gpu), segment_ids.cuda(gpu), masks.cuda(gpu)

            logits = model(inputs_id, segment_ids, masks, labels=None, mode="generate")
            output_index = get_output_index(logits, threshold=0.1)
            # types[document_id[0]] = output_index[0]
            if document_id[0] not in types:
                types[document_id[0]] = output_index[0]
            else:
                for index in output_index[0]:
                    if index not in types[document_id[0]]:
                        types[document_id[0]].append(index)
    pickle.dump(types, open(args.DATA_DIR + "/zeshel/documents/" + world + "_type", 'wb'), -1)