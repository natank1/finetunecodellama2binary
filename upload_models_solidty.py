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
    GPT2ForSequenceClassification,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)


base_output_dir='/home/ubuntu/codellama/Lora/'
output_merged_dir = "/home/ubuntu/results/news_classification_llama2_7b/final_merged_checkpoint"

if __name__ =='__main__':
    # data_yaml =get_yaml()
    path2 = "modelpath"
    new_model ="new_model"


    base_model_name = path2
    # base_model = AutoModelForCausalLM.from_pretrained(
    #     base_model_name,
    #     low_cpu_mem_usage=True,
    #     return_dict=True,
    #     torch_dtype=torch.float16,
    #     device_map=my_c.device_map,
    # )
    model = GPT2ForSequenceClassification.from_pretrained(base_model_name, output_attentions=True, output_hidden_states=True,
                                                          num_labels=2)


    # output_dir = base_output_dir+data_yaml["new_model"]+'/'
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

    # base_model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base_model_name, new_model)
    model = model.merge_and_unload()

    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    login()
    hob_model= 'name/'+new_model
    model.push_to_hub(hob_model, use_auth_token=True)
    tokenizer.push_to_hub(hob_model, use_auth_token=True)
    print ("ookkk")