import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import LlamaModel, LlamaForCausalLM, LlamaTokenizer,LlamaConfig
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
        #device = torch.device(f"cuda:{args.device_index}" if torch.cuda.is_available() else "cpu") if args['device_index'] else torch.device("cuda:0")

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
        #self.llama_model = LlamaModel._from_config(llamaconfig, torch_dtype=torch.float16,)#.to(device)
        self.llama_model = LlamaModel.from_pretrained(self.args['base_model'], load_in_8bit=False, torch_dtype=torch.float16,
                                                      local_files_only=True, cache_dir=args['cache_dir'],
                                                      device_map=self.args['device_map'])
        self.llama_model = prepare_model_for_int8_training(self.llama_model)
        self.llama_model = get_peft_model(self.llama_model, peft_config)
        self.llama_model.print_trainable_parameters()
        self.llama_model.config.use_cache = False

        self.llama_tokenizer = LlamaTokenizer.from_pretrained('D:/data/llm/model/huggyllama', use_fast=False, local_files_only=True, cache_dir=args['cache_dir'])
        #self.llama_tokenizer = LlamaTokenizer.from_pretrained(self.args['base_model'], use_fast=False, local_files_only=True, cache_dir=args['cache_dir'])
        self.llama_tokenizer.pad_token = "0"
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
            self.user_embeds = nn.Embedding.from_pretrained(self.args['user_embeds'], freeze=True)
            self.user_proj = nn.Linear(self.input_dim, self.llama_model.config.hidden_size)
        self.input_embeds = nn.Embedding.from_pretrained(self.args['input_embeds'], freeze=True)
        self.input_proj = nn.Linear(self.input_dim, self.llama_model.config.hidden_size)
        self.score = nn.Linear(self.llama_model.config.hidden_size, self.output_dim, bias=False)

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
