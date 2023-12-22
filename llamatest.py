from transformers.models.llama import LlamaModel, LlamaConfig
import torch

def run_1lama():
   llamaconfig = LlamaConfig(vocab_size=32000, hidden_size=4096//4, intermediate_size=11008//4, num_hidden_Layers=32//4, num_attention_heads=32//4, max_position_embeddings=2048//4)
   llamamodel = LlamaModel(config=llamaconfig)
   inputs_ids = torch.randint(low=0, high=llamaconfig.vocab_size, size=(4, 30))
   res = llamamodel(inputs_ids)
   print(res)

if __name__ == '__main__':
   run_1lama()
