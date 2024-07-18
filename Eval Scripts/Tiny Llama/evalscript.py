# Install necessary libraries
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Define the model and tokenizer
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def read_questions(file_path):

    """
    Reads the questions from a file and organizes them into a dictionary.

    Parameters:
    file_path (str): Path to the questions file.

    Returns:
    dict: A dictionary containing questions and their respective options.
    """
    with open(file_path, 'r') as file:
        content = file.read().strip().split('\n\n')
    questions = {}
    for i, question_block in enumerate(content):
        question_lines = question_block.strip().split('\n')
        question_text = " ".join(question_lines[:-4])
        options = {chr(65+j): question_lines[-4+j] for j in range(4)}
        questions[i+1] = {"question": question_text, "options": options}
    return questions

    """
    Prints the questions and their options.

    Parameters:
    questions (dict): A dictionary containing questions and their respective options.
    """
def print_questions(questions):
    for q_num, q_data in questions.items():
        print(f"Question {q_num}: {q_data['question']}")
        for option, text in q_data['options'].items():
            print(f"  {option}: {text}")
        print("\n" + "-"*50 + "\n")

    """
    Main function to read and print questions from a file.
    """
def main():
    questions_file = "/content/ques.txt"

    questions = read_questions(questions_file)
    print_questions(questions)

if __name__ == "__main__":
    main()

"""
    Reads the answer key from a file.

    Parameters:
    file_path (str): Path to the answer key file.

    Returns:
    dict: A dictionary containing the correct answers.
    """
def read_answer_key(file_path):
    with open(file_path, 'r') as file:
        answer_key = {}
        for line in file:
            q_num, ans = line.strip().split(' - ')
            answer_key[int(q_num)] = ans
    return answer_key

"""
    Gets predictions from the model for each question.

    Parameters:
    questions (dict): A dictionary containing questions and their respective options.

    Returns:
    dict: A dictionary containing predicted answers.
    """
def get_model_predictions(questions):
    predictions = {}
    for q_num, q_data in questions.items():
        input_text = f"Question: {q_data['question']}\nOptions:\n"
        for option, text in q_data['options'].items():
            input_text += f"{option}: {text}\n"
        input_text += "The correct answer is:"

        inputs = tokenizer(input_text, return_tensors='pt')
        outputs = model.generate(inputs['input_ids'], max_length=512)
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Improved after-process to extract only a single letter (option)
        # Assume that the model's output will be in the form "The correct answer is: A."
        predicted_answer = None
        answer_prefix = "The correct answer is: "
        start_index = output_text.find(answer_prefix)
        if start_index != -1:
            # Find the position right after the prefix
            start_index += len(answer_prefix)
            # The predicted answer should be the next character
            predicted_answer = output_text[start_index:start_index+1]

        if predicted_answer and predicted_answer in ['A', 'B', 'C', 'D']:
            predictions[q_num] = predicted_answer
        else:
            print(f"Could not predict an option for question {q_num}.")
            predictions[q_num] = "Unknown"

    return predictions

"""
    Calculates the accuracy of the model's predictions.

    Parameters:
    predictions (dict): A dictionary containing predicted answers.
    answer_key (dict): A dictionary containing the correct answers.

    Returns:
    float: The accuracy of the predictions.
    """
def calculate_accuracy(predictions, answer_key):
    correct_count = 0
    for q_num, pred in predictions.items():
        correct_ans = answer_key.get(q_num)
        if pred.strip().lower() == correct_ans.strip().lower():
            correct_count += 1
            print(f"Question {q_num}: Correct (Predicted: {pred}, Correct: {correct_ans})")
        else:
            print(f"Question {q_num}: Incorrect (Predicted: {pred}, Correct: {correct_ans})")
    accuracy = correct_count / len(predictions)
    return accuracy

answer_key = read_answer_key("/content/AIBE-18-B_Answer-Key.txt")
questions = read_questions("/content/AIBE-18-B.txt")
predictions = get_model_predictions(questions)
accuracy = calculate_accuracy(predictions, answer_key)

print(f"Model Accuracy: {accuracy:.2%}")

