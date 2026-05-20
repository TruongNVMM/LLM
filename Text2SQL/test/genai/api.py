from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Xin chào! Bạn có thể viết SQL để lấy top 5 khách hàng không? Và trong schema có bảng customers với các cột id, name, total_spent."
)

print(response.text)