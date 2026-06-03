import ollama

text = """
I work on Deep Learning, Computer Vision and Healthcare AI.
"""

prompt = f"""
Analyze the following faculty profile and extract research domain tags.

Profile:
{text}

Return only tags separated by commas.
"""

response = ollama.chat(
    model="llama3:8b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

print(response["message"]["content"])