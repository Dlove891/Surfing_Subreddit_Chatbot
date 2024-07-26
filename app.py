import numpy as np
from flask import Flask, request, render_template, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import requests
import re


# Create an app object using the flask class.
app = Flask(__name__)



@app.route('/')
def home():
    return render_template('index.html')

## What to do
#GET: a GET message is sent, and the server returns data
#POST: Used to send HTML form data to the server.
#Add: Post method to the decorator to allow for form submission
#Redirect to /predict page with the output
@app.route('/predict', methods=['POST'])
def predict():

    #your huggingface model link can go here
    API_URL = "https://api-inference.huggingface.co/models/dlove891/surfingchatbot"
    headers = {"Authorization": "Bearer Your_hf_token_here"}

    user_prompt = request.form.get('prompt')
    
    if not user_prompt:
        return render_template('index.html', prediction_text="Please enter a surfing-related question.")

    # Construct a more specific prompt
    prompt = f"Answer this surfing question in detail: {user_prompt}\nAnswer:"

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 100,  # Allow for longer responses
            "num_return_sequences": 1,
            "do_sample": True,
            "temperature": 0.2,  # Increase temperature for more variety
            "top_p": 0.95
        }
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        
        result = response.json()
        generated_text = result[0]['generated_text']
        
        # Extract only the generated answer
        response_text = generated_text.split("Answer:")[-1].strip()
        
        # Remove any text after "Question:" if present
        response_text = re.split(r'\b(?:Question:|Q:)', response_text)[0].strip()
        
        # If the response is too short, try to extract any relevant information
        if len(response_text) < 20:
            relevant_info = re.findall(r'\b(?:surf|wave|beach|board|ocean|skill|improve|practice|technique)\w*\b', generated_text, re.IGNORECASE)
            if relevant_info:
                response_text = "Related surfing concepts: " + ", ".join(relevant_info)
            else:
                response_text = "I'm not sure about that. Could you rephrase your surfing question?"
        
        # Truncate if still too long
        if len(response_text) > 200:
            response_text = response_text[:200] + "..."
        
    except requests.exceptions.RequestException as e:
        response_text = f"Error: {str(e)}"

    return render_template('index.html', prediction_text=response_text)

if __name__=='__main__':
    app.run()
