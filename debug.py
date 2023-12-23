import os
import sys
from typing import List

import fire
import pickle
import numpy as np

from utils.prompter import Prompter
from utils.data_utils import BipartiteGraphDataset, BipartiteGraphCollator, SequentialDataset, SequentialCollator
from utils.eval_utils import RecallPrecision_atK, MRR_atK, MAP_atK, NDCG_atK, AUC, getLabel

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import LlamaModel, LlamaTokenizer,LlamaConfig
from transformers.modeling_outputs import SequenceClassifierOutputWithPast

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
    set_peft_model_state_dict
)


class LLM4Rec(nn.Module):
    def __init__(self, **args):
        super(LLM4Rec, self).__init__()
        self.args = args
        self.input_dim, self.output_dim = args['input_dim'], args['output_dim']

        print(f'Initializing language decoder ...')
        # add the lora module
        peft_config = LoraConfig(
            task_type='FEATURE_EXTRACTION',
            r=self.args['lora_r'],
            lora_alpha=self.args['lora_alpha'],
            lora_dropout=self.args['lora_dropout'],
            target_modules=self.args['lora_target_modules'],
            bias='none',
        )
        llamaconfig = LlamaConfig(vocab_size=32000, hidden_size=4096//4, intermediate_size=11008//4, num_hidden_Layers=32//4, num_attention_heads=32//4, max_position_embeddings=2048//4)
        #self.llama_model = LlamaModel(config=llamaconfig).cuda()
        self.llama_model = LlamaModel._from_config(llamaconfig, torch_dtype=torch.float16).cuda()
        #self.llama_model = LlamaModel.from_pretrained('D:/data/llm/model/huggyllama', load_in_8bit=True, torch_dtype=torch.float16,
        #                                              local_files_only=True, cache_dir=args['cache_dir'],
        #                                              device_map=self.args['device_map'])
        self.llama_model = prepare_model_for_int8_training(self.llama_model)
        self.llama_model = get_peft_model(self.llama_model, peft_config)
        self.llama_model.print_trainable_parameters()
        self.llama_model.config.use_cache = False

        self.llama_tokenizer = LlamaTokenizer.from_pretrained('D:/data/llm/model/huggyllama', use_fast=False, local_files_only=True, cache_dir=args['cache_dir'])
        self.llama_tokenizer.pad_token = '0'
        self.llama_tokenizer.padding_side = "right"
        self.instruct_ids, self.instruct_mask = self.llama_tokenizer(self.args['instruction_text'][0],
                                                                     truncation=True, padding=False,
                                                                     return_tensors='pt', add_special_tokens=False).values()
        self.response_ids, self.response_mask = self.llama_tokenizer(self.args['instruction_text'][1],
                                                                     truncation=True, padding=False,
                                                                     return_tensors='pt', add_special_tokens=False).values()
        print('Language decoder initialized.')

        self.task_type = args['task_type']
        if self.task_type == 'general':
            self.user_embeds = nn.Embedding.from_pretrained(self.args['user_embeds'], freeze=True).cuda()
            self.user_proj = nn.Linear(self.input_dim, self.llama_model.config.hidden_size).cuda()
        self.input_embeds = nn.Embedding.from_pretrained(self.args['input_embeds'], freeze=True).cuda()
        self.input_proj = nn.Linear(self.input_dim, self.llama_model.config.hidden_size).cuda()
        self.score = nn.Linear(self.llama_model.config.hidden_size, self.output_dim, bias=False).cuda()

    def predict(self, inputs, inputs_mask):
        bs = inputs.shape[0]
        instruct_embeds = self.llama_model.model.embed_tokens(self.instruct_ids.cuda()).expand(bs, -1, -1)
        response_embeds = self.llama_model.model.embed_tokens(self.response_ids.cuda()).expand(bs, -1, -1)
        instruct_mask = self.instruct_mask.cuda().expand(bs, -1)
        response_mask = self.response_mask.cuda().expand(bs, -1)

        if self.task_type == 'general':
            users = self.user_proj(self.user_embeds(inputs[:, 0].unsqueeze(1)))
            items = self.input_proj(self.input_embeds(inputs[:, 1:]))
            inputs = torch.cat([users, items], dim=1)
        else:
            inputs = self.input_proj(self.input_embeds(inputs))
        inputs = torch.cat([instruct_embeds, inputs, response_embeds], dim=1)
        attention_mask = torch.cat([instruct_mask, inputs_mask, response_mask], dim=1)
        assert attention_mask.size()[0] == inputs.size()[0] and attention_mask.size()[1] == inputs.size()[1]

        outputs = self.llama_model(inputs_embeds=inputs, attention_mask=attention_mask, return_dict=True)
        pooled_output = outputs.last_hidden_state[:, -1]
        pooled_logits = self.score(pooled_output)

        return outputs, pooled_logits.view(-1, self.output_dim)

    def forward(self, inputs, inputs_mask, labels):
        outputs, pooled_logits = self.predict(inputs, inputs_mask)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(pooled_logits, labels.view(-1))

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def train(
    # model/data params
    #base_model: str = "",
    data_path: str = "",
    cache_dir: str = "",
    checkpoint_dir: str = "",
    output_dir: str = "",
    task_type: str = "",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 8,
    num_epochs: int = 1,
    learning_rate: float = 3e-4,
    cutoff_len: int = 4096,
    val_set_size: int = 0,
    lr_scheduler: str = "cosine",
    warmup_steps: int = 100,
    # lora hyperparams
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    # from peft docs: ["q_proj", "k_proj", "v_proj", "o_proj", "fc_in", "fc_out", "wte", "gate_proj", "down_proj", "up_proj"]
    lora_target_modules: List[str] = ["gate_proj", "down_proj", "up_proj"],
    # llm hyperparams
    train_on_inputs: bool = False,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca"
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Params using prompt template {prompt_template_name}:\n"
            #f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"cache_dir: {cache_dir}\n"
            f"checkpoint_dir: {checkpoint_dir}\n"
            f"output_dir: {output_dir}\n"
            f"task_type: {task_type}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lr_scheduler: {lr_scheduler}\n"
            f"warmup_steps: {warmup_steps}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"  
        )
    
    """
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    """
    gradient_accumulation_steps = batch_size // micro_batch_size

    prompter = Prompter(prompt_template_name)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
        print("gradient_accumulation_steps: ", gradient_accumulation_steps)

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    if task_type == 'general':
        dataset = BipartiteGraphDataset(data_path)
        user_embed, item_embed = (pickle.load(open(data_path + 'VanillaMF_user_embed.pkl', 'rb')).cuda(),
                                  pickle.load(open(data_path + 'VanillaMF_item_embed.pkl', 'rb')).cuda())
        item_embed = torch.cat([item_embed.mean(dim=0).unsqueeze(0), item_embed], dim=0)
        data_collator = BipartiteGraphCollator()
    elif task_type == 'sequential':
        dataset = SequentialDataset(data_path, 50)
        user_embed, item_embed = None, pickle.load(open(data_path + 'SASRec_item_embed.pkl', 'rb')).cuda()
        data_collator = SequentialCollator()
    
    #state_dict = torch.load(checkpoint_dir + 'pytorch_model.bin', map_location='cpu')
    #state_dict = {k: v.cuda() for k, v in state_dict.items() if 'lora' in k or 'user_proj' in k or 'input_proj' in k or 'score' in k}

    model = LLM4Rec(
        #base_model=base_model,
        task_type=task_type,
        cache_dir=cache_dir,
        input_dim=64,
        output_dim=dataset.m_item,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
        device_map=device_map,
        instruction_text=prompter.generate_prompt(task_type),
        user_embeds=user_embed,
        input_embeds=item_embed,
    )
    #model.load_state_dict(state_dict, strict=False)
    #del state_dict

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    # if torch.__version__ >= "2" and sys.platform != "win32":
    #     model = torch.compile(model)

    model.eval()
    topk = [1, 5, 10, 20, 100]
    results = {'Precision': np.zeros(len(topk)),
               'Recall': np.zeros(len(topk)),
               'MRR': np.zeros(len(topk)),
               'MAP': np.zeros(len(topk)),
               'NDCG': np.zeros(len(topk))}

    testData = dataset.testData
    users = np.arange(dataset.n_user)
    for u in users:
        if task_type == 'general':
            all_pos = [dataset.allPos[u]]
            groundTruth = [testData[u]]
            inputs = torch.LongTensor([u] + all_pos[0]).cuda().unsqueeze(0)
            inputs_mask = torch.ones(inputs.shape).cuda()
            _, ratings = model.predict(inputs, inputs_mask)
            exclude_index = []
            exclude_items = []
            for range_i, its in enumerate(all_pos):
                exclude_index.extend([range_i] * len(its))
                exclude_items.extend(its)
            ratings[exclude_index, exclude_items] = -(1 << 10)

        elif task_type == 'sequential':
            if len(testData[u]) == 0:
                continue
            all_pos = [testData[u][0]]
            groundTruth = [[testData[u][1]]]
            inputs = torch.LongTensor(testData[u][0]).cuda().unsqueeze(0)
            inputs_mask = torch.ones(inputs.shape).cuda()
            _, ratings = model.predict(inputs, inputs_mask)
            exclude_index = []
            exclude_items = []
            for range_i, its in enumerate(all_pos):
                exclude_index.extend([range_i] * len(its))
                exclude_items.extend(its)
            ratings[exclude_index, exclude_items] = -(1 << 10)

        _, ratings_K = torch.topk(ratings, k=topk[-1])
        ratings_K = ratings_K.cpu().numpy()

        r = getLabel(groundTruth, ratings_K)
        for j, k in enumerate(topk):
            pre, rec = RecallPrecision_atK(groundTruth, r, k)
            mrr = MRR_atK(groundTruth, r, k)
            map = MAP_atK(groundTruth, r, k)
            ndcg = NDCG_atK(groundTruth, r, k)
            results['Precision'][j] += pre
            results['Recall'][j] += rec
            results['MRR'][j] += mrr
            results['MAP'][j] += map
            results['NDCG'][j] += ndcg

    for key in results.keys():
        results[key] /= float(len(users))
    print(f'Evaluation for User: \n')
    for j, k in enumerate(topk):
        print(f'Precision@{k}: {results["Precision"][j]} \n '
              f'Recall@{k}: {results["Recall"][j]} \n '
              f'MRR@{k}: {results["MRR"][j]} \n '
              f'MAP@{k}: {results["MAP"][j]} \n '
              f'NDCG@{k}: {results["NDCG"][j]} \n')


if __name__ == "__main__":
    torch.cuda.empty_cache()
    fire.Fire(train)
