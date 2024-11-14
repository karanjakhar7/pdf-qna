import os
import tempfile
from typing import List

from core.pipeline import create_qa_chain
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel, Field


class GenerateAnswers(BaseModel):
    questions: List[str] = Field(..., title="Questions to ask the model")


app = FastAPI()


@app.post("/generate_answers")
def generate_answers(
    questions: List[str],
    pdf_file: UploadFile = File(...),
):
    if len(questions) == 0:
        return {"error": "No questions provided"}
    if len(questions) == 1:
        questions = [item.strip() for item in questions[0].split(",")]

    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, pdf_file.filename)

    try:
        # Save the uploaded file
        with open(temp_path, "wb") as f:
            f.write(pdf_file.file.read())

        qa_chain = create_qa_chain(temp_path)

        answers = qa_chain.batch(questions)

        response = []
        for q, a in zip(questions, answers):
            response.append({"question": q, "answer": a})

        return {"response": response}

    finally:
        # Clean up: remove temporary file and directory
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)
