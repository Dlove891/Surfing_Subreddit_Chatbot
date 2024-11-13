# Surfing Subreddit ChatBot

![Screen Shot 2024-07-26 at 11 49 27 AM](https://github.com/user-attachments/assets/53fce91d-bf8c-4a4d-aeda-a1e0bf29433c)

## Overview

**Please give the bot 20 seconds after submitting the first question to access the HF API properly, then try again!**

This Surf ChatBot is a full-stack web application that provides surfing-related information and answers questions about surfing. It uses a fine-tuned GPT-2 model trained on data from the r/surfing subreddit to generate responses to user queries. It is made to be replicated with any subreddit to create your own projects!

## Usage

Visit the deployed application @ https://surf-chatbot-d389c8652967.herokuapp.com/
Enter your surfing-related question in the input field and click "Get your response" to receive an AI-generated answer.

## Features

- Web-based interface for asking surfing-related questions
- AI-powered responses based on a custom-trained language model
- Responsive design for desktop and mobile devices
- Example questions to guide users
- Information about the chatbot's training data

## Tech Stack

- Frontend: HTML, CSS, JavaScript
- Backend: Python, Flask
- AI Model: GPT-2 (fine-tuned), Hugging Face Transformers
- Hosting: Heroku

## Project Structure

- `train.py`: Script for fine-tuning the GPT-2 model on surfing data and uploading it to huggingface
- `app.py`: Flask application for serving the web interface and handling API requests
- `index.html`: Main page of the web application
- `requirements.txt`: Python dependencies for the project

## Setup and Deployment

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables for API keys and tokens
4. Run locally: `python app.py`
5. Deploy to Heroku:
   - Create a new Heroku app
   - Set up the necessary buildpacks
   - Configure environment variables in Heroku
   - Deploy the code to Heroku

## Model Training

1. Data Collection:
   - Used PRAW (Python Reddit API Wrapper) to collect posts from r/Surfing subreddit
   - Focused on question posts without images, collecting up to 1000 posts
   - Gathered post titles, dates, contents, upvotes, downvotes, and top 5 comments for each post

2. Data Preprocessing:
   - Combined title, content, and comments into a single text field
   - Cleaned the text by removing extra spaces, URLs, and special characters
   - Split the data into training and validation sets (80/20 split)

3. Model and Tokenizer Setup:
   - Used the pre-trained GPT-2 model and tokenizer from Hugging Face
   - Set the pad token to the EOS token for proper padding

4. Dataset Preparation:
   - Created a custom TextDataset class for handling the tokenized data
   - Used DataCollatorForLanguageModeling for efficient batching

5. Training Configuration:
   - Set up TrainingArguments with the following key settings:
     - 1000 training epochs
     - Batch size of 2 with gradient accumulation steps of 6
     - Learning rate of 5e-5 with linear scheduler and warmup steps
     - Evaluation and model saving every 1000 steps
     - Used eval_loss as the metric for selecting the best model

6. Fine-tuning:
   - Used the Hugging Face Trainer for fine-tuning the model
   - Trained on the prepared dataset with the specified configuration

7. Model Saving and Uploading:
   - Saved the fine-tuned model and tokenizer locally
   - Uploaded the model to Hugging Face Hub for easy deployment and inference

## Future Improvements

- Expand the training dataset for more comprehensive knowledge
- Implement user feedback mechanisms for continuous improvement
- Add multi-language support for international surfers

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For any questions or feedback, please open an issue on this repository.
