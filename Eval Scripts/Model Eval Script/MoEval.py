pip install boto3 torch transformers tqdm SentencePiece

import boto3
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from torch.cuda.amp import autocast
import gc

def download_files_from_s3(bucket_name, prefix, local_directory):
    session = boto3.Session()
    s3 = session.client('s3')
    
    if not os.path.exists(local_directory):
        os.makedirs(local_directory)
    
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        for file in page.get('Contents', []):
            file_name = file['Key']
            local_path = os.path.join(local_directory, os.path.relpath(file_name, prefix))
            
            local_file_directory = os.path.dirname(local_path)
            if not os.path.exists(local_file_directory):
                os.makedirs(local_file_directory)
            
            print(f"Downloading {file_name}...")
            s3.download_file(bucket_name, file_name, local_path)
            print(f"Saved to {local_path}")

def read_questions(file_path):
    with open(file_path, 'r') as file:
        content = file.read().strip().split('\n')
    
    questions = {}
    current_question = None
    current_options = {}
    
    for line in content:
        line = line.strip()
        if not line:
            continue
        
        if line[0].isdigit():
            if current_question:
                questions[current_question['number']] = current_question
            
            question_number = int(line.split('.')[0])
            question_text = line
            current_question = {'number': question_number, 'question': question_text, 'options': {}}
            current_options = {}
        elif line[0] == '(' and ')' in line:
            option_letter, option_text = line.split(')', 1)
            option_letter = option_letter.strip('(').strip()
            option_text = option_text.strip()
            current_options[option_letter] = option_text
            current_question['options'] = current_options
    
    if current_question:
        questions[current_question['number']] = current_question
    
    return questions

def read_answer_key(file_path):
    with open(file_path, 'r') as file:
        return {int(line.split('-')[0].strip()): line.split('-')[1].strip() for line in file}

def load_model(model_directory):
    print(f"Loading model from {model_directory}")
    print("Contents of model directory:")
    for file in os.listdir(model_directory):
        print(f"  {file}")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(model_directory)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_directory)
        print("Tokenizer loaded successfully")
    except Exception as e:
        print(f"Error loading tokenizer: {str(e)}")
        raise

    return model, tokenizer

def get_model_predictions(model, tokenizer, questions, device):
    predictions = {}
    batch_size = 1  # Process one question at a time to minimize memory usage

    try:
        model.to(device)
        model.eval()
        
        for q_num, q_data in tqdm(questions.items(), desc="Generating predictions"):
            input_text = f"{q_data['question']}\n"
            for option, text in q_data['options'].items():
                input_text += f"{option}) {text}\n"
            input_text += "The correct answer is:"
            
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
            
            try:
                inputs = inputs.to(device)
                
                with torch.no_grad(), autocast(enabled=True):
                    outputs = model.generate(**inputs, max_new_tokens=5, num_return_sequences=1)
                
                predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                predicted_answer = predicted_text.split("The correct answer is:")[-1].strip()[0]
                
                predictions[q_num] = predicted_answer
                
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"WARNING: GPU OOM for question {q_num}. Falling back to CPU.")

                    model.to('cpu')
                    inputs = inputs.to('cpu')
                    
                    with torch.no_grad():
                        outputs = model.generate(**inputs, max_new_tokens=5, num_return_sequences=1)
                    
                    predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    predicted_answer = predicted_text.split("The correct answer is:")[-1].strip()[0]
                    
                    predictions[q_num] = predicted_answer
                    
                    model.to(device)
                else:
                    raise e
            
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    
    except Exception as e:
        print(f"An error occurred during prediction: {str(e)}")
        raise

    return predictions

def calculate_accuracy(predictions, answer_key):
    correct_count = 0
    for q_num, pred in predictions.items():
        correct_ans = answer_key.get(q_num)
        if pred.strip().upper() == correct_ans.strip().upper():
            correct_count += 1
            print(f"Question {q_num}: Correct (Predicted: {pred}, Correct: {correct_ans})")
        else:
            print(f"Question {q_num}: Incorrect (Predicted: {pred}, Correct: {correct_ans})")
    accuracy = correct_count / len(predictions)
    return accuracy

bucket_name = "sagemaker-us-east-1-637423474134"
prefix = "k-llama3-8b-fullds-lrcos-r32-q-2024-07-04-22-11-56-336/output/model/"
    
local_directory = "/tmp/downloaded_model"
model_directory = local_directory  # The model will be loaded from where it's downloaded

download_files_from_s3(bucket_name, prefix, local_directory)


def main():
    model, tokenizer = load_model(model_directory)
    
    try:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

        if torch.cuda.is_available():
            device = torch.device("cuda")
            torch.cuda.set_per_process_memory_fraction(0.9)
        else:
            device = torch.device("cpu")
        print(f"Using device: {device}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

    try:
        questions = read_questions("./AIBE-18-B.txt")
        print(f"Successfully read {len(questions)} questions")
        if len(questions) == 0:
            raise ValueError("No questions were read from the file")
    except Exception as e:
        print(f"Error reading questions: {str(e)}")
        return

    try:
        answer_key = read_answer_key("./AIBE-18-B_Answer-Key.txt")
        print(f"Successfully read {len(answer_key)} answers")
        if len(answer_key) == 0:
            raise ValueError("No answers were read from the file")
    except Exception as e:
        print(f"Error reading answer key: {str(e)}")
        return

    predictions = get_model_predictions(model, tokenizer, questions, device)

    accuracy = calculate_accuracy(predictions, answer_key)
    print(f"Model Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    main()


