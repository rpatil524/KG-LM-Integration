
import numpy as np
import pandas as pd
import jsonlines
import gc
import torch
from torch import nn

from IPython.display import display, clear_output

from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel
from transformers.activations import GELUActivation
from transformers.modeling_outputs import MaskedLMOutput
from transformers import DataCollatorForWholeWordMask
from datasets import load_from_disk, load_dataset
from transformers import BertTokenizer, DistilBertTokenizer
from transformers.data.data_collator import _torch_collate_batch
import evaluate
from transformers import PreTrainedModel


import torch
import torch.nn as nn
import torch.nn.functional as F

# the KIM desined in this study
class WordItgtor(nn.Module):
    
    #embed_dim_lm is the embedding length of language model, which is 768
    #embed_dim_kg is the embedding length of entities, which is 200
    def __init__(self, embed_dim_lm, embed_dim_kg):
        super(WordItgtor, self).__init__()
        
        self.tt_embed_dim = embed_dim_lm, embed_dim_kg
        
        self.fc_kg = nn.Linear(embed_dim_kg,embed_dim_lm)
        self.fc_lm = nn.Linear(embed_dim_lm,embed_dim_lm)
        self.fc1 = nn.Linear(embed_dim_lm * 2, embed_dim_lm * 4)
        self.fc2 = nn.Linear(embed_dim_lm * 4, embed_dim_lm * 2)
        self.fc3 = nn.Linear(embed_dim_lm * 2, embed_dim_lm)
    
    def forward(self, x_lm, x_kg, kg_mask=None):        
        
        x_kg = self.fc_kg(x_kg)
        x_lm = self.fc_lm(x_lm)
        
        x = torch.cat((x_lm,x_kg),dim=-1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

class WordItgtor_alt(nn.Module):
    

    def __init__(self, embed_dim_lm, embed_dim_kg):
        super(WordItgtor_alt, self).__init__()
        
        self.tt_embed_dim = embed_dim_lm, embed_dim_kg
        
        self.fc_kg = nn.Linear(embed_dim_kg,embed_dim_lm)
        self.fc_lm = nn.Linear(embed_dim_lm,embed_dim_lm)
        self.fc1 = nn.Linear(embed_dim_lm * 2, embed_dim_lm * 4)
        self.fc2 = nn.Linear(embed_dim_lm * 4, embed_dim_lm * 2)
        self.fc3 = nn.Linear(embed_dim_lm * 2, embed_dim_lm)
        self.multihead_attn = nn.MultiheadAttention(embed_dim_lm, 4,batch_first=True)
        

    
    
    def forward(self, x_lm, x_kg, kg_mask=None):        

        
        x_kg = self.fc_kg(x_kg)
        x_lm = self.fc_lm(x_lm)
        
        x = torch.cat((x_lm,x_kg),dim=-1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        res = self.multihead_attn(x,x,x)
        
        return res[0]

class BERTModified(nn.Module): #PreTrainedModel
    def __init__(self, bert_model_name, base_model, config, kg_size =46685):
        
#         super().__init__(config) #For PreTrainedModel
        super().__init__() ## for nn.Module
        
        self.base_model = base_model
        self.config = config
        self.kg_size = kg_size
                
        self.activation = GELUActivation() # for distilbert
        self.vocab_transform = nn.Linear(self.config.dim, self.config.dim)
        self.vocab_layer_norm = nn.LayerNorm(self.config.dim, eps=1e-12)
        self.vocab_projector = nn.Linear(self.config.dim, self.config.vocab_size)
        
        self.kg_projector = nn.Linear(self.config.dim, self.kg_size)
        
        self.mlm_loss_fct = nn.CrossEntropyLoss()
        
        ## set to eval
        self.base_model.eval()
        
        ## freeze model
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        ## initialization of integrator
        self.itgt = WordItgtor(self.config.dim,200)

    def forward(
        self,
        kg_embedding = None,           ## given
        kg_embedding_mask = None,      ## given
        kg_embedding_mask_qid = None,      ## given
        kg_embedding_mask_index = None,
        input_ids = None,              ## given
        attention_mask = None,         ## given
        head_mask = None,
        inputs_embeds = None,
        labels = None,                 ## given
        output_attentions = None,
        output_hidden_states = None,
        return_dict= None,):
        
        base_model_output = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        

        
        ## Get LM embedding
        hidden_states = base_model_output[0]  # (bs, seq_length, dim)
        

        
        ## Use hidden_states and kg_embedding and perform INTEGRATION
        kg_embedding = torch.tensor(kg_embedding).to(device='cuda:0')
        kg_embedding_mask_index = torch.tensor(kg_embedding_mask_index).to(device='cuda:0')
        
        prediction_logits = self.itgt(hidden_states, kg_embedding)
        prediction_logits = self.activation(prediction_logits)  # (bs, seq_length, dim)
        prediction_logits = self.vocab_layer_norm(prediction_logits)  # (bs, seq_length, dim)
    
        lm_prediction_logits = self.vocab_projector(prediction_logits)  # (bs, seq_length, vocab_size)
        kg_prediction_logits = self.kg_projector(prediction_logits)  # (bs, seq_length, kg_size)

        mlm_loss = None
        kg_loss = None
        

        
        if labels is not None:
            mlm_loss = self.mlm_loss_fct(lm_prediction_logits.view(-1, lm_prediction_logits.size(-1)), labels.view(-1))
        if kg_embedding_mask_index is not None:
            kg_loss = self.mlm_loss_fct(kg_prediction_logits.view(-1, kg_prediction_logits.size(-1)), kg_embedding_mask_index.view(-1))

        total_loss = mlm_loss + kg_loss
        
        return MaskedLMOutput(
            loss=total_loss,
            logits=lm_prediction_logits,
            hidden_states=base_model_output.hidden_states,
            attentions=base_model_output.attentions,
        )
    
    
    
class BERTModified_KG(nn.Module): #PreTrainedModel
    def __init__(self, bert_model_name, base_model, config, kg_size =46685):
        
#         super().__init__(config) #For PreTrainedModel
        super().__init__() ## for nn.Module
        
        self.base_model = base_model
        self.config = config
        self.kg_size = kg_size
                
        self.activation = GELUActivation() # for distilbert
        self.vocab_transform = nn.Linear(self.config.dim, self.config.dim)
        self.vocab_layer_norm = nn.LayerNorm(self.config.dim, eps=1e-12)
        self.vocab_projector = nn.Linear(self.config.dim, self.config.vocab_size)
        
        self.kg_projector = nn.Linear(self.config.dim, self.kg_size)
        
        self.mlm_loss_fct = nn.CrossEntropyLoss()
        
        ## set to eval
        self.base_model.eval()
        
        ## freeze model
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        ## initialization of integrator
        self.itgt = WordItgtor(self.config.dim,200)

    def forward(
        self,
        kg_embedding = None,           ## given
        kg_embedding_mask = None,      ## given
        kg_embedding_mask_qid = None,      ## given
        kg_embedding_mask_index = None,
        input_ids = None,              ## given
        attention_mask = None,         ## given
        head_mask = None,
        inputs_embeds = None,
        labels = None,                 ## given
        output_attentions = None,
        output_hidden_states = None,
        return_dict= None,):
        
        base_model_output = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        

        
        ## Get LM embedding
        hidden_states = base_model_output[0]  # (bs, seq_length, dim)
        

        
        ## Use hidden_states and kg_embedding and perform INTEGRATION
        kg_embedding = torch.tensor(kg_embedding).to(device='cuda:0')
        kg_embedding_mask_index = torch.tensor(kg_embedding_mask_index).to(device='cuda:0')
        
        prediction_logits = self.itgt(hidden_states, kg_embedding)
        prediction_logits = self.activation(prediction_logits)  # (bs, seq_length, dim)
        prediction_logits = self.vocab_layer_norm(prediction_logits)  # (bs, seq_length, dim)
    
        lm_prediction_logits = self.vocab_projector(prediction_logits)  # (bs, seq_length, vocab_size)
        kg_prediction_logits = self.kg_projector(prediction_logits)  # (bs, seq_length, kg_size)

        mlm_loss = None
        kg_loss = None
        

        
        if labels is not None:
            mlm_loss = self.mlm_loss_fct(lm_prediction_logits.view(-1, lm_prediction_logits.size(-1)), labels.view(-1))
        if kg_embedding_mask_index is not None:
            kg_loss = self.mlm_loss_fct(kg_prediction_logits.view(-1, kg_prediction_logits.size(-1)), kg_embedding_mask_index.view(-1))

        total_loss = mlm_loss + kg_loss
        
        return MaskedLMOutput(
            loss=total_loss,
            logits=kg_prediction_logits,
            hidden_states=base_model_output.hidden_states,
            attentions=base_model_output.attentions,
        )
    
    
    
class BERTModified_KGRaw(nn.Module): #PreTrainedModel
    def __init__(self, bert_model_name, base_model, config,kg_size =46685):
        
#         super().__init__(config) #For PreTrainedModel
        super().__init__() ## for nn.Module
        
        self.base_model = base_model
        self.config = config
        self.kg_size =kg_size 
                
        self.activation = GELUActivation() # for distilbert
        
        self.kg_transform = nn.Linear(200, self.config.dim)
        
        self.vocab_transform = nn.Linear(self.config.dim, self.config.dim)
        self.vocab_layer_norm = nn.LayerNorm(self.config.dim, eps=1e-12)
        self.vocab_projector = nn.Linear(self.config.dim, self.config.vocab_size)
        
        self.kg_projector = nn.Linear(self.config.dim, self.kg_size)
        
        self.mlm_loss_fct = nn.CrossEntropyLoss()
        
        ## set to eval
        self.base_model.eval()
        
        ## freeze model
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        ## initialization of integrator
        self.itgt = WordItgtor(self.config.dim,200)

    def forward(
        self,
        kg_embedding = None,           ## given
        kg_embedding_mask = None,      ## given
        kg_embedding_mask_qid = None,      ## given
        kg_embedding_mask_index = None,
        input_ids = None,              ## given
        attention_mask = None,         ## given
        head_mask = None,
        inputs_embeds = None,
        labels = None,                 ## given
        output_attentions = None,
        output_hidden_states = None,
        return_dict= None,):
        
        base_model_output = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = base_model_output[0]  # (bs, seq_length, dim)
        
        kg_embedding = torch.tensor(kg_embedding).to(device='cuda:0')
        kg_embedding_mask_index = torch.tensor(kg_embedding_mask_index).to(device='cuda:0')
        

        prediction_logits = self.kg_transform(kg_embedding)  # (bs, seq_length, 200)
        prediction_logits = self.activation(prediction_logits)  # (bs, seq_length, dim)
        prediction_logits = self.vocab_layer_norm(prediction_logits)  # (bs, seq_length, dim)
        prediction_logits = self.kg_projector(prediction_logits)  # (bs, seq_length, vocab_size)
        
        
        kg_prediction_logits = prediction_logits
        
        kg_loss = None
        

        if kg_embedding_mask_index is not None:
            kg_loss = self.mlm_loss_fct(kg_prediction_logits.view(-1, kg_prediction_logits.size(-1)), kg_embedding_mask_index.view(-1))


        total_loss = kg_loss
        
        return MaskedLMOutput(
            loss=total_loss,
            logits=kg_prediction_logits,
            hidden_states=base_model_output.hidden_states,
            attentions=base_model_output.attentions,
        )
    

class BERTModified_LMRaw(nn.Module): #PreTrainedModel
    def __init__(self, bert_model_name, base_model, config, kg_size =46685):
        
#         super().__init__(config) #For PreTrainedModel
        super().__init__() ## for nn.Module
        
        self.base_model = base_model
        self.config = config
        self.kg_size =kg_size
                
        self.activation = GELUActivation() # for distilbert
        self.vocab_transform = nn.Linear(self.config.dim, self.config.dim)
        self.vocab_layer_norm = nn.LayerNorm(self.config.dim, eps=1e-12)
        self.vocab_projector = nn.Linear(self.config.dim, self.config.vocab_size)
        
        self.kg_projector = nn.Linear(self.config.dim, self.kg_size)
        
        self.mlm_loss_fct = nn.CrossEntropyLoss()
        
        ## set to eval
        self.base_model.eval()
        
        ## freeze model
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        ## initialization of integrator
        self.itgt = WordItgtor(self.config.dim,200)

    def forward(
        self,
        kg_embedding = None,           ## given
        kg_embedding_mask = None,      ## given
        kg_embedding_mask_qid = None,      ## given
        kg_embedding_mask_index = None,
        input_ids = None,              ## given
        attention_mask = None,         ## given
        head_mask = None,
        inputs_embeds = None,
        labels = None,                 ## given
        output_attentions = None,
        output_hidden_states = None,
        return_dict= None,):
        
        base_model_output = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        ## Get LM embedding
        hidden_states = base_model_output[0]  # (bs, seq_length, dim)
        

        

        prediction_logits = self.vocab_transform(hidden_states)  # (bs, seq_length, dim)
        prediction_logits = self.activation(prediction_logits)  # (bs, seq_length, dim)
        prediction_logits = self.vocab_layer_norm(prediction_logits)  # (bs, seq_length, dim)
        prediction_logits = self.vocab_projector(prediction_logits)  # (bs, seq_length, vocab_size)
    

        lm_prediction_logits = prediction_logits


        mlm_loss = None

        

        
        if labels is not None:
            mlm_loss = self.mlm_loss_fct(lm_prediction_logits.view(-1, lm_prediction_logits.size(-1)), labels.view(-1))

        total_loss = mlm_loss
        
        return MaskedLMOutput(
            loss=total_loss,
            logits=lm_prediction_logits,
            hidden_states=base_model_output.hidden_states,
            attentions=base_model_output.attentions,
        )
    

class BERTModified_alt(nn.Module): #PreTrainedModel
    def __init__(self, bert_model_name, base_model, config,kg_size =46685):
        
#         super().__init__(config) #For PreTrainedModel
        super().__init__() ## for nn.Module
        
        self.base_model = base_model
        self.config = config
        self.kg_size = kg_size
                
        self.activation = GELUActivation() # for distilbert
        self.vocab_transform = nn.Linear(self.config.dim, self.config.dim)
        self.vocab_layer_norm = nn.LayerNorm(self.config.dim, eps=1e-12)
        self.vocab_projector = nn.Linear(self.config.dim, self.config.vocab_size)
        
        self.kg_projector = nn.Linear(self.config.dim, self.kg_size)
        
        self.mlm_loss_fct = nn.CrossEntropyLoss()
        
        ## set to eval
        self.base_model.eval()
        
        ## freeze model
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        ## initialization of integrator
        self.itgt = WordItgtor_alt(self.config.dim,200)

    def forward(
        self,
        kg_embedding = None,           ## given
        kg_embedding_mask = None,      ## given
        kg_embedding_mask_qid = None,      ## given
        kg_embedding_mask_index = None,
        input_ids = None,              ## given
        attention_mask = None,         ## given
        head_mask = None,
        inputs_embeds = None,
        labels = None,                 ## given
        output_attentions = None,
        output_hidden_states = None,
        return_dict= None,):
        
        base_model_output = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        

        
        ## Get LM embedding
        hidden_states = base_model_output[0]  # (bs, seq_length, dim)
        

        
        ## TODO: Use hidden_states and kg_embedding and perform INTEGRATION
        
        kg_embedding = torch.tensor(kg_embedding).to(device='cuda:0')
        kg_embedding_mask_index = torch.tensor(kg_embedding_mask_index).to(device='cuda:0')
        
        prediction_logits = self.itgt(hidden_states, kg_embedding)
        prediction_logits = self.activation(prediction_logits)  # (bs, seq_length, dim)
        prediction_logits = self.vocab_layer_norm(prediction_logits)  # (bs, seq_length, dim)


    
        lm_prediction_logits = self.vocab_projector(prediction_logits)  # (bs, seq_length, vocab_size)
        kg_prediction_logits = self.kg_projector(prediction_logits)  # (bs, seq_length, kg_size)

        mlm_loss = None
        kg_loss = None
        

        
        if labels is not None:
            mlm_loss = self.mlm_loss_fct(lm_prediction_logits.view(-1, lm_prediction_logits.size(-1)), labels.view(-1))
        if kg_embedding_mask_index is not None:
            kg_loss = self.mlm_loss_fct(kg_prediction_logits.view(-1, kg_prediction_logits.size(-1)), kg_embedding_mask_index.view(-1))

        total_loss = mlm_loss + kg_loss
        
        return MaskedLMOutput(
            loss=total_loss,
            logits=lm_prediction_logits,
            hidden_states=base_model_output.hidden_states,
            attentions=base_model_output.attentions,
        )

    
    
    
class BERTModified_altKG(nn.Module): #PreTrainedModel
    def __init__(self, bert_model_name, base_model, config, kg_size =46685):
        
#         super().__init__(config) #For PreTrainedModel
        super().__init__() ## for nn.Module
        
        self.base_model = base_model
        self.config = config
        self.kg_size = kg_size 
                
        self.activation = GELUActivation() # for distilbert
        self.vocab_transform = nn.Linear(self.config.dim, self.config.dim)
        self.vocab_layer_norm = nn.LayerNorm(self.config.dim, eps=1e-12)
        self.vocab_projector = nn.Linear(self.config.dim, self.config.vocab_size)
        
        self.kg_projector = nn.Linear(self.config.dim, self.kg_size)
        
        self.mlm_loss_fct = nn.CrossEntropyLoss()
        
        ## set to eval
        self.base_model.eval()
        
        ## freeze model
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        ## initialization of integrator
        self.itgt = WordItgtor_alt(self.config.dim,200)

    def forward(
        self,
        kg_embedding = None,           ## given
        kg_embedding_mask = None,      ## given
        kg_embedding_mask_qid = None,      ## given
        kg_embedding_mask_index = None,
        input_ids = None,              ## given
        attention_mask = None,         ## given
        head_mask = None,
        inputs_embeds = None,
        labels = None,                 ## given
        output_attentions = None,
        output_hidden_states = None,
        return_dict= None,):
        
        base_model_output = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        ## Get LM embedding
        hidden_states = base_model_output[0]  # (bs, seq_length, dim)
        

        
        ## TODO: Use hidden_states and kg_embedding and perform INTEGRATION
        
        kg_embedding = torch.tensor(kg_embedding).to(device='cuda:0')
        kg_embedding_mask_index = torch.tensor(kg_embedding_mask_index).to(device='cuda:0')
        
        prediction_logits = self.itgt(hidden_states, kg_embedding)
        prediction_logits = self.activation(prediction_logits)  # (bs, seq_length, dim)
        prediction_logits = self.vocab_layer_norm(prediction_logits)  # (bs, seq_length, dim)

    
        lm_prediction_logits = self.vocab_projector(prediction_logits)  # (bs, seq_length, vocab_size)
        kg_prediction_logits = self.kg_projector(prediction_logits)  # (bs, seq_length, kg_size)

        mlm_loss = None
        kg_loss = None
        

        
        if labels is not None:
            mlm_loss = self.mlm_loss_fct(lm_prediction_logits.view(-1, lm_prediction_logits.size(-1)), labels.view(-1))
        if kg_embedding_mask_index is not None:
            kg_loss = self.mlm_loss_fct(kg_prediction_logits.view(-1, kg_prediction_logits.size(-1)), kg_embedding_mask_index.view(-1))

        total_loss = mlm_loss + kg_loss
        
        return MaskedLMOutput(
            loss=total_loss,
            logits=kg_prediction_logits,
            hidden_states=base_model_output.hidden_states,
            attentions=base_model_output.attentions,
        )
