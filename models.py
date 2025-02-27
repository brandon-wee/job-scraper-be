import os
from supabase import create_client, Client
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from prompts import Prompts
from bs4 import BeautifulSoup
import requests
import re
import json
from dotenv import load_dotenv
import hmac
import hashlib
from typing import List
import base64
import io
import hmac, hashlib
from tfidf import compute_tf_idf_similarity
import PyPDF2
import pandas as pd

load_dotenv()

# Login to Supabase
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

# Pydantic Schema
class SkillItem(BaseModel):
    name: str = Field(..., description="The name of the skill.")
    description: str = Field(..., description="A brief explanation of the skill, its context, and proficiency level if available.")

class JobDetailsSchema(BaseModel):
    position: str = Field(..., description="The position of the job")
    company: str = Field(..., description="The company name")
    skills_required: List[SkillItem] = Field(..., description="A list of skills (technical, soft or business) required for the job")
    experience: str = Field(..., description="The prior work experience required for the job")
    location: str = Field(..., description="The country where the job listing is located")
    
class JobDetailsExtractLLM:
    # Initialize Google Generative AI (gemini-2.0-flash-lite)
    def __init__(self):
        self.parser = JsonOutputParser(pydantic_object=JobDetailsSchema)
        prompt = PromptTemplate(
            template=Prompts.SYSTEM_PROMPT,
            input_variables=["jobHTML"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
        )
        model = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash-lite", api_key=os.getenv("GOOGLE_API_KEY"))

        self.chain = prompt | model 
    
    def add_user(self, user_id: str):
        hashed_id = hash_id(user_id)
        try:
            response = supabase.table("USER") \
                .insert({"user_id": hashed_id, "user_name": ""}) \
                .execute()
            return {"success": True, "error": None}
        except Exception as e:
            return {"success": False, "error": "User already exists."}

    def save_job_details(self, message, user_id: str):
        try:
            soup = BeautifulSoup(message["jobDetailsDiv"], 'html.parser')
            job_contents = soup.get_text(separator="\n")

            job_url = message["jobURL"]
            job_id = job_url.split("currentJobId=")[1].split("&")[0]
            hashed_id = hash_id(user_id)

            # Check if user exists
            response = supabase.table("USER") \
                .select("user_id") \
                .eq("user_id", hashed_id) \
                .execute()
            
            if len(response.data) == 0:
                return {"success": False, "error": "User does not exist."}

            # Check if this job has already been saved by the user
            response = supabase.table("USER_JOB") \
                .select("job_id") \
                .eq("job_id", job_id) \
                .eq("user_id", hashed_id) \
                .execute()

            if len(response.data) > 0:
                return {"success": False, "error": "Job already saved."}
            
            # Check if this job has already been saved by another user
            response = supabase.table("JOB") \
                .select("job_id") \
                .eq("job_id", job_id) \
                .execute()
            
            if len(response.data) == 0:

                response = self.chain.invoke(job_contents)
                obj = json.loads(re.sub(r"```json|```", "", json.dumps(response.content, indent=4)).strip())
                obj = self.parser.parse(obj)
                position = obj["position"]
                company = obj["company"]
                skills_required = self.convert_skills_list(obj["skills_required"])
                experience = obj["experience"]
                address, lat, long = self.get_company_address(company, obj["location"])

                # Check if company exists in the database
                response = supabase.table("COMPANY") \
                    .select("company_id") \
                    .eq("company_name", company) \
                    .eq("company_address", address) \
                    .execute()
                
                if len(response.data) == 0:
                    # Insert new company
                    response = supabase.table("COMPANY") \
                        .insert({"company_name": company, "company_address": address, "company_lat": lat, "company_long": long}) \
                        .execute()
                    
                company_id = response.data[0]["company_id"]
                # Insert job details
                response = supabase.table("JOB") \
                    .insert({"job_id": job_id, "job_title": position, "company_id": int(company_id), "job_skills_required": skills_required, "job_experience_level": experience, "job_url": job_url}) \
                    .execute()
            
            # Insert user job
            response = supabase.table("USER_JOB") \
                .insert({"job_id": job_id, "user_id": hashed_id, "application_status": "Saved"}) \
                .execute()

            return {"success": True, "error": None}
        
        except Exception as e:
            print(e)
            return {"success": False, "error": str(e)}


    def get_company_address(self, company, country):
        url = "https://maps.googleapis.com/maps/api/place/findplacefromtext/json"

        params = {
            "input": company + " " + country,
            "inputtype": "textquery",
            "fields": "formatted_address,geometry",
            "key": os.getenv("PLACES_API_KEY")
        }
        response = requests.get(url, params=params)
        if response.status_code == 200 and response.json()["status"] != "ZERO_RESULTS":
            data = response.json()
            formatted_address = data["candidates"][0]["formatted_address"]
            latitude = data["candidates"][0]["geometry"]["location"]["lat"]
            longitude = data["candidates"][0]["geometry"]["location"]["lng"]
            
            return formatted_address, float(latitude), float(longitude)
        else:
            return "Not Found", None, None
    
    def convert_skills_list(self, skills):
        print(skills)
        if skills == []:
            return ""
        return '\n'.join(["- " + skill['name'] + ": " + skill["description"] for skill in skills])

# ----------------- Resume Skills Similarity -----------------

def decode_pdf(pdf_data: str) -> bytes:
    """Decodes JSON Base64 string back into PDF bytes."""
    pdf_bytes = base64.b64decode(pdf_data)  # Decode Base64 back to bytes
    return io.BytesIO(pdf_bytes)

def hash_id(user_id):
    return hmac.new(os.getenv("HASH_SECRET").encode(), user_id.encode(), hashlib.sha256).hexdigest()


class SkillItem(BaseModel):
    name: str = Field(..., description="The name of the skill.")
    description: str = Field(..., description="A brief explanation of the skill, its context, and proficiency level if available.")

class ExtractorSchema(BaseModel):
    technical_skills: List[SkillItem] = Field(..., description="A list of technical skills, including programming languages, tools, and frameworks.")
    # soft_skills: List[SkillItem] = Field(..., description="A list of soft skills, including leadership, communication, teamwork, etc.")
    # domain_expertise: List[SkillItem] = Field(..., description="A list of domain-specific knowledge areas such as finance, healthcare, cybersecurity, etc.")

class SkillsSchema(BaseModel):
    compatible_skills: List[str] = Field(..., description="A list of the name of the skills that are compatible with the job requirements.")
    missing_skills: List[str] = Field(..., description="A list of the name of the skills that are missing from the job requirements.")

class ResumeSkillsSimilarity:

    def __init__(self):
        self.huggingface_auth = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}"}
        self.huggingface_url = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
        model = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash", api_key=os.getenv("GOOGLE_API_KEY"))

        self.extractor_parser = JsonOutputParser(pydantic_object=ExtractorSchema)
        self.extractor_prompt = PromptTemplate(
            template=Prompts.EXTRACTOR_SYSTEM_PROMPT,
            input_variables=["resume_text"],
            partial_variables={"format_instructions": self.extractor_parser.get_format_instructions()}
        )
        self.extractor_chain = self.extractor_prompt | model

        self.skills_parser = JsonOutputParser(pydantic_object=SkillsSchema)
        self.skills_prompt = PromptTemplate(
            template=Prompts.SKILLS_SYSTEM_PROMPT,
            input_variables=["resume_skills", "job_skills"],
            partial_variables={"format_instructions": self.skills_parser.get_format_instructions()}
        )
        self.skills_chain = self.skills_prompt | model

    def get_similarity(self, resume_contents, user_id):
        job_listings = self.get_job_listings(user_id)
        resume_skills = self.extract_skills_from_pdf(decode_pdf(resume_contents))
        result = []
        for job in job_listings:
            job_skills = job['job_skills_required']
            skills = self.extract_compatible_and_missing_skills(resume_skills, job_skills)
            similarity = self.cosine_similarity(resume_skills, job_skills, skills["compatible_skills"], skills["missing_skills"])
            
            result.append({"position": job['job_title'], 
                           "company": job['company_name'], 
                           "similarity_score": similarity, 
                           "compatible_skills": '\n'.join([f"- {skill}" for skill in skills['compatible_skills']]), 
                           "missing_skills": '\n'.join([f"- {skill}" for skill in skills['missing_skills']])
                           })

        return result

    def cosine_similarity(self, resume_skills, job_skills, compatible_skills, missing_skills):
        """Compute cosine similarity between two sentences using a Sentence Transformer."""
        # Compute cosine similarity
        response = requests.post(self.huggingface_url, headers=self.huggingface_auth, json={
            "inputs": {
                "source_sentence": resume_skills,
                "sentences": [job_skills]
            }
        })
        bert_similarity = response.json()[0]
        tfidf_similarity = (compute_tf_idf_similarity(resume_skills, job_skills) + 1) / 2
        compatibility_similarity = len(compatible_skills) / (len(compatible_skills) + len(missing_skills))
        weights = [0.3, 0.2, 0.5]
        X = [bert_similarity, tfidf_similarity, compatibility_similarity]
        similarity = sum([X[i] * weights[i] for i in range(len(X))])
        return similarity

    def extract_text_from_pdf(self, resume_contents):
        """Extract text from a PDF file."""
        reader = PyPDF2.PdfReader(resume_contents)
        text = ''
        for page in reader.pages:
            text += page.extract_text() + '\n'
        
        return text

    def extract_skills_from_pdf(self, resume_contents):
        """Extract skills from a PDF file using Google Generative AI."""
        text = self.extract_text_from_pdf(resume_contents)
        processed_text = json.loads(re.sub(r"```json|```", "", json.dumps(self.extractor_chain.invoke(text).content, indent=4)))
        result = self.extractor_parser.parse(processed_text)['technical_skills']

        return self.convert_skills_list(result)
    
    def convert_skills_list(self, skills):
        return '\n'.join(["- " + skill['name'] + ": " + skill["description"] for skill in skills])

    def get_job_listings(self, user_id):
        hashed_id = hash_id(user_id)
        response = supabase.table("JOB").select("job_title, COMPANY!inner(company_name), job_skills_required, USER_JOB!inner(user_id)").eq("USER_JOB.user_id", hashed_id).execute()
        df = pd.DataFrame(response.data)
        df.drop(columns=["USER_JOB"], inplace=True)
        df["company_name"] = df["COMPANY"].apply(lambda x: x["company_name"])
        df.drop(columns=["COMPANY"], inplace=True)
        return df.to_dict(orient='records')
    
    def extract_compatible_and_missing_skills(self, resume_skills, job_skills):
        response = self.skills_chain.invoke({"resume_skills": resume_skills, "job_skills": job_skills})
        processed_text = json.loads(re.sub(r"```json|```", "", json.dumps(response.content, indent=4)))
        # print(processed_text)
        result = self.skills_parser.parse(processed_text)
        return result
