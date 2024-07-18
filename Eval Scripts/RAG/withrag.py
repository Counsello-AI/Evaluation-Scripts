import requests
import json

# Set the URL and headers for the request
url = 'your-url'
headers = {
    'Content-Type': 'application/json'
}
data = {
    'userPrompt': 'Which of the following procedures is dealt under Section 164-A of the Code of Criminal Procedure, 1973?(A) Medical examination of the victim of rape.(B) Attendance of witness by police officer.(C) Recording of confession statement.(D) Recording of first information report by police officer.'}
# Make the POST request with streaming
response = requests.post(url, headers=headers, json=data, stream=True)

# Check if the response status is OK
if response.status_code == 200:
    # Initialize a string to store the complete response
    full_response = ""

    # Process the response stream in real-time
    for line in response.iter_lines():
        if line:
            decoded_line = line.decode('utf-8')
            if decoded_line.startswith('data: '):
                event_data = decoded_line[6:]  # Remove the 'data: ' prefix
                try:
                    json_data = json.loads(event_data)
                    if 'delta' in json_data and 'text' in json_data['delta']:
                        text = json_data['delta']['text']
                        print(text, end='', flush=True)  # Print the text in real-time
                        full_response += text  # Append the text to the full response
                except json.JSONDecodeError:
                    # Handle JSON decode error if any
                    pass

    # Print the full response at the end
    # print("\nFull Response: ", full_response)
else:
    print(f"Failed to connect, status code: {response.status_code}")

