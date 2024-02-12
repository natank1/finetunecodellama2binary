import transformers
import huggingface_hub as hf
from huggingface_hub import login
# Load the trained model and tokenizer
model_name="model-NAME"
model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
login()
hob_model = 'NAME/'+model_name+'_0'
model.push_to_hub(hob_model, use_auth_token=True)
tokenizer.push_to_hub(hob_model, use_auth_token=True)
