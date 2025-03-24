from flask import Flask, render_template, request, jsonify
import google.generativeai as genai


# Configure Google Gemini API
genai.configure(api_key="AIzaSyArYlTXDE_K98U-B292wjYad1YIv3ys_1s")

app = Flask(__name__)

# Function to process user query
def get_gemini_response(user_input):
    model = genai.GenerativeModel("gemini-1.5-pro")
    system_instruction = (
        "You are an AI expert in e-waste management. "
        "Provide responses focused on responsible e-waste disposal, recycling, and environmental impact."
    )
    response = model.generate_content(system_instruction + "\nUser: " + user_input)
    return response.text

@app.route('/')
def index():
    return render_template('index.html')

    
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['message']
    response_text = get_gemini_response(user_input)
    return jsonify({"response": response_text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,debug=True)
