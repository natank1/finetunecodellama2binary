# https://towardsdatascience.com/fine-tune-your-own-llama-2-model-in-a-colab-notebook-df9823a04a32
from  utils import get_yaml
import train_const as my_c
import torch
import os
from peft import LoraConfig, PeftModel,AutoPeftModelForCausalLM
from huggingface_hub import login
# login()
from peft import AutoPeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoModelForSequenceClassification,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)


base_output_dir='path1'
output_merged_dir = "final_merged_checkpoint"

if __name__ =='__main__':
    data_yaml =get_yaml()
    new_model = data_yaml["new_model"]


    base_model_name = data_yaml["base_model"]
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map=my_c.device_map,output_scores=True)


    # base_model =     AutoModelForSequenceClassification.from_pretrained(
    #     base_model_name,
    #     num_labels=2,
    #     low_cpu_mem_usage=True,
    #     return_dict=True,
    #     torch_dtype=torch.float16,
    #     device_map=my_c.device_map
    # )
    # output_dir = base_output_dir+data_yaml["new_model"]+'/'
    tokenizer = AutoTokenizer.from_pretrained(data_yaml["new_tok_file"], trust_remote_code=True)

    # base_model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base_model, new_model)
    model = model.merge_and_unload()

    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    login()
    hob_model= 'name/'+new_model
    model.push_to_hub(hob_model, use_auth_token=True)
    tokenizer.push_to_hub(hob_model, use_auth_token=True)
    print ("ookkk")