import json
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()

def extract_cv_data(pdf_path):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    with open(pdf_path, "rb") as pdf_file:
        uploaded_file = client.files.create(
            file=pdf_file,
            purpose="user_data"
        )

        response = client.responses.create(model="gpt-5.6-luna",
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_file",
                        "file_id": uploaded_file.id
                    },
                    {
                        "type": "input_text",
                        "text": """
Extract the candidate's work experience and education from this CV.

Return only the information that is actually present in the CV.
Do not invent or guess missing information.

For work experience, extract:
- company
- position
- from
- to
- description

For education, extract:
- school
- degree
- from
- to
- field
"""
                    }
                ]
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "cv_data",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "work_experience": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "company": {"type": "string"},
                                    "position": {"type": "string"},
                                    "from": {"type": "string"},
                                    "to": {"type": "string"},
                                    "description": {"type": "string"}
                                },
                                "required": [
                                    "company",
                                    "position",
                                    "from",
                                    "to",
                                    "description"
                                ],
                                "additionalProperties": False
                            }
                        },
                        "education": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "school": {"type": "string"},
                                    "degree": {"type": "string"},
                                    "from": {"type": "string"},
                                    "to": {"type": "string"},
                                    "field": {"type": "string"}
                                },
                                "required": [
                                    "school",
                                    "degree",
                                    "from",
                                    "to",
                                    "field"
                                ],
                                "additionalProperties": False
                            }
                        }
                    },
                    "required": [
                        "work_experience",
                        "education"
                    ],
                    "additionalProperties": False
                }
            }
        }
    )

    return json.loads(response.output_text)
            
def main():
    cv_folder = Path("cv")

    pdf_files = list(cv_folder.glob("*.pdf"))

    if not pdf_files:
        print("Neboli nájdené žiadne PDF súbory v priečinku 'cv'.")
        return

    for pdf_path in pdf_files:

        result = extract_cv_data(pdf_path)

        output_file = cv_folder / f"{pdf_path.stem}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                result,
                f,
                indent=4,
                ensure_ascii=False
            )

    
        print("Hotovo → output.json")


if __name__ == "__main__":
    main()
