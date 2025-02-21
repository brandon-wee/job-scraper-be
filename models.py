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

load_dotenv()

# Login to Supabase
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

# Pydantic Schema
class JobDetailsSchema(BaseModel):
    position: str = Field(..., description="The position of the job")
    company: str = Field(..., description="The company name")
    technical_requirements: str = Field(..., description="The technical requirements for the job")
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
        model = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash-lite-preview-02-05", api_key=os.getenv("GOOGLE_API_KEY"))

        self.chain = prompt | model 
        

    def save_job_details(self, message: str, user_id: str):
        with open("job.html", "w", encoding="utf-8") as f:
            f.write(message["jobDetailsDiv"])

        soup = BeautifulSoup(message["jobDetailsDiv"], 'html.parser')
        job_contents = soup.get_text(separator="\n")

        response = self.chain.invoke(job_contents)
        obj = json.loads(re.sub(r"```json|```", "", json.dumps(response.content, indent=4)).strip())
        obj = self.parser.parse(obj)
        # obj = response

        obj["location"] = self.get_company_address(obj["company"], obj["location"])
        obj["hash_user_details"] = self.hash_id(user_id)
        obj["url"] = message["jobURL"]

        obj["job_id"] = obj["url"].split("currentJobId=")[1].split("&")[0]

        try:
            supabase.table("job_details").insert(obj).execute()
            return {"success": True, "error": None}
        
        except Exception as e:
            print("Error:", e)
            return {"success": False, "error": str(e)}


    def get_company_address(self, company, country):
        url = "https://maps.googleapis.com/maps/api/place/findplacefromtext/json"

        params = {
            "input": company + " " + country,
            "inputtype": "textquery",
            "fields": "formatted_address",
            "key": os.getenv("PLACES_API_KEY")
        }

        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            return data["candidates"][0]["formatted_address"]
        else:
            return "Not Found"
    

    def hash_id(self, user_id):
        return hmac.new(os.getenv("HASH_SECRET").encode(), user_id.encode(), hashlib.sha256).hexdigest()