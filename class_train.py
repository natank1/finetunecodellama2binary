import datasets
from scipy.special import softmax

from torch.utils.data import   DataLoader
from random import shuffle
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorWithPadding,
)

import train_const as my_c
from peft import LoraConfig, TaskType
from trl import SFTTrainer
from lora_pre_prco import get_bnb_config

from torch.utils.data import Dataset
import torch



class TextClassificationDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx],

        }

def get_data(input_data):
    shuffle(input_data)
    new_dic = { "text": [i[0] for i in input_data],"labels": [i[1] for i in input_data]}
    data0 = datasets.Dataset.from_dict(new_dic)
    return data0

class MyTrainer(SFTTrainer):

    def compute_loss(self, model, inputs,  return_outputs=False):

        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs["labels"]
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss
def get_tokenizer(model_name):
    # Load LLaMA tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer

def prepare_obj(tokenizer,dataset):
    inputs = tokenizer([i for i in dataset["text"]], padding=True, truncation=True, return_tensors="pt")
    labels = torch.tensor(dataset['labels'])
    text_obj = TextClassificationDataset(input_ids =inputs['input_ids'],attention_mask= inputs['attention_mask'],labels=labels )

    return text_obj
if __name__== '__main__':
    train_data= [("Hi I am happy to hear you",0),("it is good",1), ("write list com",1),("While (True",0),("I won ",1),("if a>b",0) ]
    dataset = get_data(train_data)
    model_name = "codellama/CodeLlama-7b-hf"
    tokenizer =get_tokenizer(model_name)
    text_loader= prepare_obj(tokenizer, dataset)
    del dataset
    new_model = "tunedmodel"
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    bnb_config = get_bnb_config()
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,num_labels=2,
        quantization_config=bnb_config,
        device_map=my_c.device_map,output_scores=True)
    model.config.use_cache = False

    # Load LoRA configuration
    peft_config = LoraConfig(
        lora_alpha=my_c.lora_alpha,
        lora_dropout=my_c.lora_dropout,
        r=my_c.lora_r,
        bias="none",
        task_type=TaskType.SEQ_CLS ,
    )

    # Set training parameters
    training_arguments = TrainingArguments(
        output_dir=my_c.output_dir,
        num_train_epochs=my_c.num_train_epochs,
        per_device_train_batch_size=my_c.per_device_train_batch_size,
        gradient_accumulation_steps=my_c.gradient_accumulation_steps,
        optim=my_c.optim,
        save_steps=my_c.save_steps,
        logging_steps=my_c.logging_steps,
        learning_rate=my_c.learning_rate,
        weight_decay=my_c.weight_decay,
        fp16=my_c.fp16,
        bf16=my_c.bf16,
        max_grad_norm=my_c.max_grad_norm,
        max_steps=my_c.max_steps,
        warmup_ratio=my_c.warmup_ratio,
        group_by_length=my_c.group_by_length,
        lr_scheduler_type=my_c.lr_scheduler_type,
        report_to="tensorboard"
    )
    print ("Go traininfgaaa")
    print (new_model)

    model.config.pad_token_id = model.config.eos_token_id
    trainer = MyTrainer(
        model=model,
        args=training_arguments,
        max_seq_length=my_c.max_seq_length,
        tokenizer=tokenizer,
        train_dataset=text_loader,
        dataset_text_field="text",
        peft_config=peft_config,
        data_collator=data_collator,
    )



    # Train model
    print ("Go traininfg")
    trainer.train()

    trainer.save_model(new_model)
    print ("Training is over")
    trainer=[]
    model =[]
    print ("Start eval")
    print ("Load model")
    model_test = AutoModelForSequenceClassification.from_pretrained(
        new_model ,
        device_map="auto",
        num_labels=2,
        quantization_config=bnb_config)
    test_dataset = get_data(train_data)
    text_loader = prepare_obj(tokenizer, test_dataset)


    with torch.no_grad():
        tr_data = DataLoader(text_loader, batch_size=1, shuffle=False)
        print("loading 1")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for step, batch in enumerate(tr_data):
            batch = tuple(batch[t].to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            test_output = model_test(b_input_ids,
                                attention_mask=b_input_mask,
                                labels=b_labels)
            pr = softmax(test_output.logits[0].cpu().numpy())

    print ("over")
