import torch.nn as nn
import torch
from pytorch_transformers.modeling_bert import (
    BertPreTrainedModel,
    BertConfig,
    BertModel,
)
from torch.nn.init import xavier_uniform_ as xavier_uniform
import torch.nn.functional as F




class SoftEmbedding(nn.Module):
    def __init__(self, 
                wte: nn.Embedding,
                n_tokens: int = 10, 
                random_range: float = 0.5,
                initialize_from_vocab: bool = True):
        """appends learned embedding to 
        Args:
            wte (nn.Embedding): original transformer word embedding
            n_tokens (int, optional): number of tokens for task. Defaults to 10.
            random_range (float, optional): range to init embedding (if not initialize from vocab). Defaults to 0.5.
            initialize_from_vocab (bool, optional): initalizes from default vocab. Defaults to True.
        """
        super(SoftEmbedding, self).__init__()
        self.wte = wte
        self.n_tokens = n_tokens
        self.learned_embedding = nn.parameter.Parameter(self.initialize_embedding(wte,
                                                                               n_tokens, 
                                                                               random_range, 
                                                                               initialize_from_vocab))
            
    def initialize_embedding(self, 
                             wte: nn.Embedding,
                             n_tokens: int = 10, 
                             random_range: float = 0.5, 
                             initialize_from_vocab: bool = True):
        """initializes learned embedding
        Args:
            same as __init__
        Returns:
            torch.float: initialized using original schemes
        """
        if initialize_from_vocab:
            return self.wte.weight[:n_tokens].clone().detach()
        return torch.FloatTensor(n_tokens, wte.weight.size(1)).uniform_(-random_range, random_range)
            
    def forward(self, tokens):
        """run forward pass
        Args:
            tokens (torch.long): input tokens before encoding
        Returns:
            torch.float: encoding of text concatenated with learned task specifc embedding
        """
        input_embedding = self.wte(tokens[:, self.n_tokens:])
        learned_embedding = self.learned_embedding.repeat(input_embedding.size(0), 1, 1)
        return torch.cat([learned_embedding, input_embedding], 1)


def js_div(p_output, q_output, get_softmax=True):
    """
    Function that measures JS divergence between target and output logits:
    """
    KLDivLoss = nn.KLDivLoss(reduction='batchmean')
    if get_softmax:
        p_output = F.softmax(p_output)
        q_output = F.softmax(q_output)
    log_mean_output = ((p_output + q_output )/2).log()
    return (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output))/2


class Bert(nn.Module):

    def __init__(self, args):
        super(Bert, self).__init__()

        self.name = args.model
        self.gpu = args.gpu
        self.loss = nn.BCEWithLogitsLoss(reduction="mean")
        self.bert = BertModel.from_pretrained(args.bert_dir)
        # self.types = torch.load(args.DATA_DIR + "/crowd/types.pt").cuda(self.gpu)
        self.dropout=nn.Dropout(0.1)
        self.final = nn.Linear(1024, 10331)
        xavier_uniform(self.final.weight)
        self.prompt= SoftEmbedding(self.bert.embeddings.word_embeddings,n_tokens=20)
        self.bert.embeddings.word_embeddings=self.prompt
        # self.jsd=js_div
        

    def forward(self, inputs_id, segment_ids, masks, labels, mode):
        
        inputs_id=torch.cat([torch.full((inputs_id.size(0),self.prompt.n_tokens), 50256,dtype=torch.long).to(inputs_id.device), inputs_id], 1)
        masks=torch.cat([torch.full((masks.size(0),self.prompt.n_tokens), 1,dtype=torch.long).to(inputs_id.device), masks], 1)
        segment_ids=torch.cat([torch.full((segment_ids.size(0),self.prompt.n_tokens), 0,dtype=torch.long).to(inputs_id.device), segment_ids], 1)        
        x, out_pooler = self.bert(inputs_id, segment_ids, attention_mask=masks)
        # alpha = torch.nn.functional.softmax(torch.matmul(self.types, x.transpose(1, 2)), dim=2)
        # m = alpha.matmul(x)
        # logits = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)
        out_pooler=self.dropout(out_pooler)
        logits=self.final(out_pooler)
        if mode != "generate":
            loss = self.loss(logits, labels.float())
            return logits, loss
        else:
            return logits


def pick_model(args):
    if args.model == "bert":
        model = Bert(args)
    else:
        raise RuntimeError("wrong model name")
    if args.gpu >= 0:
        model.cuda(args.gpu)
    return model