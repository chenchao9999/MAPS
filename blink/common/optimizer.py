# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import os
import numpy as np

from pytorch_transformers.modeling_bert import (
    BertPreTrainedModel,
    BertConfig,
    BertModel,
)
from pytorch_transformers.tokenization_bert import BertTokenizer
from torch import nn

from pytorch_transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_transformers.optimization import AdamW


patterns_optimizer = {
    'additional_layers': ['additional'],
    'top_layer': ['additional', 'bert_model.encoder.layer.11.'],
    'top4_layers': [
        'additional',
        'bert_model.encoder.layer.11.',
        'encoder.layer.10.',
        'encoder.layer.9.',
        'encoder.layer.8',
    ],
    'all_encoder_layers': ['final', 'final2', 'additional', 'bert_model.encoder.layer', 'category_transformer1', 'linking_transformer1', 'category_transformer2', 'linking_transformer2'],
    'all': ['final', 'final2', 'additional', 'bert_model.encoder.layer', 'bert_model.embeddings', 'category_transformer1', 'linking_transformer1', 'category_transformer2', 'linking_transformer2'],
}

def get_bert_optimizer(models, type_optimization, learning_rate, fp16=False):
    """ Optimizes the network with AdamWithDecay
    """
    if type_optimization not in patterns_optimizer:
        print(
            'Error. Type optimizer must be one of %s' % (str(patterns_optimizer.keys()))
        )
    parameters_with_decay = []
    parameters_without_decay = []
    parameters_with_decay_names = []
    parameters_without_decay_names = []
    no_decay = ['bias', 'gamma', 'beta']
    patterns = patterns_optimizer[type_optimization]
    # unfreeze_layers = ['layer.7', 'layer.8', 'layer.9','layer.10', 'layer.11', 'layer.12', 'encoder.pooler', 'out.']

    for model in models:
        for n, p in model.named_parameters():
            # p.requires_grad = any(ele in n for ele in unfreeze_layers)
            if p.requires_grad:
                if any(t in n for t in patterns):
                    if any(t in n for t in no_decay):
                        parameters_without_decay_names.append(n)
                        parameters_without_decay.append(p)
                    else:
                        parameters_with_decay_names.append(n)
                        parameters_with_decay.append(p)
                    # if "type_encoder" in n :    
                    #     if "embedding" in n :
                    #         p.requires_grad=True
                    #     else:
                    #         p.requires_grad=False

    print('The following parameters will be optimized WITH decay:')
    print(ellipse(parameters_with_decay_names, 5, ' , '))
    print('The following parameters will be optimized WITHOUT decay:')
    print(ellipse(parameters_without_decay_names, 5, ' , '))

    optimizer_grouped_parameters = [
        {'params': parameters_with_decay, 'weight_decay': 0.01},
        {'params': parameters_without_decay, 'weight_decay': 0.0}
        # {'params': [p for n, p in model.named_parameters() if  "src_" in n or "final" in n ], 'weight_decay': 0.01, 'lr': 3e-3}
        
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate,
        correct_bias=False
    )

    return optimizer


def ellipse(lst, max_display=5, sep='|'):
    """
    Like join, but possibly inserts an ellipsis.
    :param lst: The list to join on
    :param int max_display: the number of items to display for ellipsing.
        If -1, shows all items
    :param string sep: the delimiter to join on
    """
    # copy the list (or force it to a list if it's a set)
    choices = list(lst)
    # insert the ellipsis if necessary
    if max_display > 0 and len(choices) > max_display:
        ellipsis = '...and {} more'.format(len(choices) - max_display)
        choices = choices[:max_display] + [ellipsis]
    return sep.join(str(c) for c in choices)
