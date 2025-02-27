from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from models import JobDetailsExtractLLM, ResumeSkillsSimilarity

app = Flask(__name__)
CORS(app)
job_extrator = JobDetailsExtractLLM()
similarity_checker = ResumeSkillsSimilarity()

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
    
@app.route('/get_similarity', methods=['POST', 'GET'])
def get_similarity():
    try:
        contents = request.get_json()
        resume_contents = contents['resume_contents']
        user_id = contents['user_id']

        result = similarity_checker.get_similarity(resume_contents, user_id)
        return jsonify({"result": result})
    
    except Exception as e:
        print("Error:", e)
        return jsonify({"success": False, "error": str(e), "similarity": 0, "resume_skills": ""})

@app.route('/add_user', methods=['POST', 'GET'])
def add_user():
    try:
        contents = request.get_json()
        user_id = contents['user_id']

        result = job_extrator.add_user(user_id)
        return jsonify(result)
    
    except Exception as e:
        print("Error:", e)
        return jsonify({"success": False, "error": str(e)})