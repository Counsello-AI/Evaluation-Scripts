# Installing the necessary packages

!pip install openai
!pip install openai==0.28

# Import the OpenAI library
import openai

# Set the API key for OpenAI
openai.api_key = "your-api-key"

# Function to read questions from a file

def read_questions(file_path):
    with open(file_path, 'r') as file:
        content = file.read().strip().split('\n\n')
    questions = {}
    for i, question_block in enumerate(content):
        question_lines = question_block.strip().split('\n')
        question_text = " ".join(question_lines[:-4])
        options = {chr(65+j): question_lines[-4+j] for j in range(4)}
        questions[i+1] = {"question": question_text, "options": options}
    return questions

# Function to read the answer key from a file

def read_answer_key(file_path):
    with open(file_path, 'r') as file:
        answer_key = {}
        for line in file:
            q_num, ans = line.strip().split(' - ')
            answer_key[int(q_num)] = ans
    return answer_key

# Function to get model predictions for the questions

def get_model_predictions(questions):
    predictions = {}
    for q_num, q_data in questions.items():

        # Prepare the input text for the model
        input_text = f"Question: {q_data['question']}\nOptions:\n"
        for option, text in q_data['options'].items():
            input_text += f"{option}: {text}\n"
        input_text += "The correct answer is:"

        # Get the model response
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": input_text}
            ],
            max_tokens=50,
            temperature=0
        )

        # Extract the predicted answer directly from the output
        predicted_answer = response.choices[0].message['content'].strip()

        # Store the prediction
        predictions[q_num] = predicted_answer

    return predictions

# Function to calculate the accuracy of the model predictions

def calculate_accuracy(predictions, answer_key):
    correct_count = 0
    total_questions = len(predictions)

    for q_num, pred in predictions.items():
        correct_ans = answer_key.get(q_num)

        if pred is not None and correct_ans is not None:
            # Extract only the option letter from the predicted answer
            pred_option = pred.split(':')[0].strip()

            # Only compare if both predicted and correct answer are not None
            if pred_option.lower() == correct_ans.strip().lower():
                correct_count += 1
                print(f"Question {q_num}: Correct (Predicted: {pred_option}, Correct: {correct_ans})")
            else:
                print(f"Question {q_num}: Incorrect (Predicted: {pred_option}, Correct: {correct_ans})")
        else:
            print(f"Question {q_num}: Could not predict an option or no correct answer available.")

    # Calculate the accuracy
    accuracy = correct_count / total_questions
    return accuracy

# Read the answer key and questions from the provided file paths
answer_key = read_answer_key("/content/AIBE-18-B_Answer-Key.txt")
questions = read_questions("/content/AIBE-18-B.txt")

# Get the model predictions
predictions = get_model_predictions(questions)

# Calculate the accuracy of the predictions
accuracy = calculate_accuracy(predictions, answer_key)

# Print the model accuracy
print(f"Model Accuracy: {accuracy:.2%}")