from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from models import JobDetailsExtractLLM

app = Flask(__name__)
CORS(app)
job_extrator = JobDetailsExtractLLM()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/save', methods=['POST', 'GET'])
def save():
    try:
        contents = request.get_json()
        message = contents['message']
        user_id = contents['userId']

        result = job_extrator.save_job_details(message, user_id)
        return jsonify(result)
    
    except Exception as e:
        print("Error:", e)
        return jsonify({"success": False, "error": str(e)})
    