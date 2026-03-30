from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()
OPENAI_MODEL = "gpt-5.4"