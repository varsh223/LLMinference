# llminterface
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from intel_extension_for_transformers.neural_chat.config import (
    ModelArguments,
    DataArguments,
    FinetuningArguments,
    TextGenerationFinetuningConfig,
)
from intel_extension_for_transformers.neural_chat.chatbot import finetune_model, build_chatbot, PipelineConfig
from intel_extension_for_transformers.transformers import MixedPrecisionConfig

# Step 1: Fine-tune the model
# Load and save the base model and tokenizer locally (run this in an environment with internet access)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model.save_pretrained("./local-llama-2-7b")
tokenizer.save_pretrained("./local-llama-2-7b")

# Fine-tune the model using the local path
model_args = ModelArguments(model_name_or_path="./local-llama-2-7b")
data_args = DataArguments(train_file="alpaca_data.json", validation_split_percentage=1)
training_args = TrainingArguments(
    output_dir='./tmp',
    do_train=True,
    do_eval=True,
    num_train_epochs=3,
    overwrite_output_dir=True,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    save_strategy="no",
    log_level="info",
    save_total_limit=2,
 bf16=True,
)
finetune_args = FinetuningArguments()
finetune_cfg = TextGenerationFinetuningConfig(
    model_args=model_args,
    data_args=data_args,
    training_args=training_args,
    finetune_args=finetune_args,
)

# Fine-tune the model
finetune_model(finetune_cfg)

# Step 2: Use the fine-tuned model to build a chatbot
# Assuming the fine-tuned model is saved in the same output directory
finetuned_model_path = "./tmp"

# Configure and build the chatbot using the fine-tuned model
config = PipelineConfig(optimization_config=MixedPrecisionConfig())
chatbot = build_chatbot(config=config, model_name_or_path=finetuned_model_path)

# Function to ensure the chatbot answers only academic queries
def academic_query(query):
    response = chatbot.predict(query=query)
    if is_academic_response(response):
        return response
    else:
        return "This chatbot is designed to answer academic queries only."
def is_academic_response(response):
    # Simple check: return True if the response contains certain academic keywords.
    academic_keywords = ["theory", "research", "study", "science", "explain", "define", "process", "history"]
    return any(keyword in response.lower() for keyword in academic_keywords)
    # Make a prediction using the chatbot
query = "Tell me about the latest research in artificial intelligence."
response = academic_query(query)

# Print the response
print(response)
