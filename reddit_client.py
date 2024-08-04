import praw
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

os.environ["HF_TOKEN"] = ""

model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Function to generate a response using the model
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs.input_ids, max_length=150, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Replace the following with your Reddit app credentials
reddit = praw.Reddit(
    client_id='client_id',
    client_secret='client_secret',
    user_agent='ChangeMeClient/0.1 by kajol_m'
)

for submission in reddit.subreddit("cricket").hot(limit=10):
    print(submission.title)
    for top_level_comment in submission.comments:
        if isinstance(top_level_comment, praw.models.MoreComments):
            continue
        print(top_level_comment.body)
