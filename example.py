import base64
import requests

# Load the image and encode it to base64
with open(r'C:\Users\user\Desktop\content\api_project\1000016.jpg', 'rb') as img_file:
    base64_image = base64.b64encode(img_file.read()).decode('utf-8')

# Create the payload
payload = {
    "base64": base64_image
}

# Send the POST request
response = requests.post('http://127.0.0.1:8000/predict', json=payload)

# Print raw response content
print("Response Status Code:", response.status_code)
print("Response Content:", response.text)

# Try parsing JSON if content is present
try:
    print(response.json())
except requests.exceptions.JSONDecodeError:
    print("Failed to decode JSON from response.")
