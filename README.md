# Surfing Subreddit Finetuned GPT2 ChatBot

![Screen Shot 2024-07-26 at 11 49 27 AM](https://github.com/user-attachments/assets/53fce91d-bf8c-4a4d-aeda-a1e0bf29433c)

## Overview

This Surf ChatBot is a full-stack web application that provides surfing-related information and answers questions about surfing. It uses a fine-tuned GPT-2 model trained on data from the r/surfing subreddit to generate responses to user queries.

## Usage

Visit the deployed application @ https://surf-chatbot-d389c8652967.herokuapp.com/
Enter your surfing-related question in the input field and click "Get your response" to receive an AI-generated answer.
**Please give the bot 20 seconds after submitting the first question to boot up, then try again!**

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

- `train.py`: Script for fine-tuning the GPT-2 model on surfing data
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

The chatbot uses a GPT-2 model fine-tuned on surfing-related content from Reddit. The training process involved:

1. Data collection from r/surfing using PRAW
2. Data preprocessing and cleaning
3. Fine-tuning GPT-2 using Hugging Face Transformers
4. Uploading the model to Hugging Face Hub for inference

## Future Improvements

- Expand the training dataset for more comprehensive knowledge
- Implement user feedback mechanisms for continuous improvement
- Add multi-language support for international surfers

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For any questions or feedback, please open an issue on this repository.
