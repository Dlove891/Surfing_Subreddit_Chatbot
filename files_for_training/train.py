import pandas as pd
import praw
from datetime import datetime
import re

# Can request for personal reddit api usage info on https://old.reddit.com/prefs/apps/
# Click create a new app
reddit = praw.Reddit(client_id='Your client ID here',
                     client_secret='Your client Secret here',
                     user_agent='Your reddit username here') 

# Select any subreddit you want. I chose surfing.
subreddit = reddit.subreddit("Surfing")

titles = []
dates = []
contents = []
upvotes = []
downvotes = []
comments_list = []

# Collect up to 1000 posts, there may be a limit depending on where you get the data from.
# new, top, hot all will provide different quatitites. 
# This call sequence only grabbed 275 posts. 
post_limit = 1000
batch_size = 100

# Use after method to bypass reddit rate limits. 
after = None

def is_question(title):
    return title.strip().endswith('?')

def has_image(post):
    return any(media in post.url.lower() for media in ['.jpg', '.jpeg', '.png', '.gif', 'i.imgur.com'])

# Main reddit post data retreival loop
while len(titles) < post_limit:
    # Fetch the next batch of posts
    posts = list(subreddit.new(limit=batch_size, params={'after': after}))

    if not posts:
        break  # Break if no posts are returned (end of subreddit or rate limit reached)

    for post in posts:
        if is_question(post.title) and not has_image(post):
            titles.append(post.title)
            dates.append(datetime.fromtimestamp(post.created_utc).strftime('%Y-%m-%d %H:%M:%S'))
            contents.append(post.selftext)
            upvotes.append(post.ups)
            downvotes.append(post.downs)

            # Fetch top 10 comments
            post.comments.replace_more(limit=None)
            comments = [comment.body for comment in post.comments[:5]]
            comments_list.append(comments)

    # Update 'after' to the fullname of the last post in the current batch
    after = posts[-1].fullname

    print(f"Collected {len(titles)} posts so far...")

# Create DataFrame
df = pd.DataFrame({
    'Title': titles,
    'Date': dates,
    'Contents': contents,
    'Upvotes': upvotes,
    'Downvotes': downvotes,
    'Comments': comments_list
})

print(f"Collected {len(df)} posts in total")


from sklearn.model_selection import train_test_split

# Feature engineering
df["Text"] = df["Title"] + " " + df["Contents"] + " " + df["Comments"].apply(lambda x: " ".join(x))

# Clean the text:
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    # text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    return text

df['Text'] = df['Text'].apply(clean_text)

train_texts, val_texts = train_test_split(df['Text'], test_size=0.2, random_state=21)

from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import torch
# Load pre-trained GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')


# Set pad_token to eos_token
tokenizer.pad_token = tokenizer.eos_token


# Tokenize the texts
def tokenize_texts(texts, tokenizer):
    return tokenizer('\n\n'.join(texts.tolist()), return_tensors='pt', max_length=512, truncation=True, padding=True)

train_encodings = tokenize_texts(train_texts, tokenizer)
val_encodings = tokenize_texts(val_texts, tokenizer)

# Prepare datasets
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return self.encodings.input_ids.size(0)

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

train_dataset = TextDataset(train_encodings)
val_dataset = TextDataset(val_encodings)

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Set up training arguments
training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=1000,  # You can increase if needed
    per_device_train_batch_size=2,  # Adjust according to your GPU memory
    gradient_accumulation_steps=6,  # Accumulate gradients to simulate a larger batch size
    save_steps=1000,  # Save more frequently to capture the best model
    save_total_limit=5,  # Keep more checkpoints
    evaluation_strategy="steps",  # Evaluate at every save step
    eval_steps=1000,  # Evaluate every 1000 steps
    load_best_model_at_end=True,  # Load the best model at the end of training
    metric_for_best_model="eval_loss",  # Use validation loss as the metric for early stopping
    greater_is_better=False,  # Lower loss is better
    learning_rate=5e-5,  # Starting learning rate
    weight_decay=0.01,  # Add weight decay for regularization
    lr_scheduler_type="linear",  # Use a linear scheduler for learning rate
    warmup_steps=500,  # Warmup steps for the learning rate scheduler
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Fine-tune the model
trainer.train()

# Saving the model to memory in /models
output_dir = './models'
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Pushing your model to huggingface. 
from huggingface_hub import HfApi, HfFolder, Repository, create_repo, upload_folder
from huggingface_hub import login
login('Your hf token goes here')



# Create a new repository
repo_id = "your_username/your_new_model_name"
create_repo(repo_id, exist_ok=True)

# Upload the model and tokenizer to the repository
upload_folder(
    folder_path='./models',
    repo_id=repo_id,
    commit_message="Initial model upload"
)




