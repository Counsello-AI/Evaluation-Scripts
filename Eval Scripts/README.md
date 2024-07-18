# Evaluation-Scripts

# AI Model Predictions

This repository contains three Python scripts for AI model predictions using different libraries and APIs. These scripts demonstrate how to read questions from a file, get predictions from AI models, and calculate the accuracy of the predictions.

## Installation

Before running the scripts, you need to install the required packages. Use the following commands to install them:

```bash
pip install openai requests transformers torch
```

Scripts
1. OpenAI API Predictions:

This script reads questions from a file, sends them to the OpenAI API for predictions, and calculates the accuracy of the predictions.

Installation:
```bash
pip install openai
```

Usage
> Set your OpenAI API key in the script.
> Prepare your questions and answer key files.
> Run the script
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

2. Transformers and PyTorch:
   
This script reads questions from a file, uses a pre-trained model from the Hugging Face library to get predictions, and calculates the accuracy of the predictions.

Installation:
```bash
pip install torch transformers
```
Usage
> Download and prepare your questions and answer key files.
> Run the script.
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

3. Streaming Data with Requests:
   
This script sends a POST request with streaming to an endpoint and processes the response in real-time.

Installation:

```bash
pip install requests
```

Usage
> Set the URL and data for your request.
> Run the script.
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

License
This project is licensed under the MIT License - see the LICENSE file for details.

```bash

This README file provides a comprehensive guide for installing dependencies, understanding, and running each script. Make sure to replace `"your-api-key"` with your actual OpenAI API key.

```
