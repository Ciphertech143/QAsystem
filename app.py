from fastapi import FastAPI, Form, Request, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForSeq2SeqLM
from transformers import pipeline
import PyPDF2
import os
import aiofiles
import csv
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# MobileBERT specific models
qa_tokenizer = AutoTokenizer.from_pretrained("mrm8488/mobilebert-uncased-finetuned-squadv1")
qa_model = AutoModelForQuestionAnswering.from_pretrained("mrm8488/mobilebert-uncased-finetuned-squadv1")
qa_pipeline = pipeline('question-answering', model=qa_model, tokenizer=qa_tokenizer)

qg_tokenizer = AutoTokenizer.from_pretrained("valhalla/t5-small-qg-hl")
qg_model = AutoModelForSeq2SeqLM.from_pretrained("valhalla/t5-small-qg-hl")
qg_pipeline = pipeline('text2text-generation', model=qg_model, tokenizer=qg_tokenizer)

def file_processing(file_path):
    logger.info(f"Processing file: {file_path}")
    # Load data from PDF
    with open(file_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        # Process each page of the PDF
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            yield page_text

def generate_questions(page_text):
    # Generate questions for a given page using T5 QG model
    questions = qg_pipeline(page_text, max_length=50, num_return_sequences=1)
    return [question['generated_text'].strip() for question in questions]

def generate_answers(page_text, questions):
    # Generate answers for a given page and questions using MobileBERT QA model
    answers = []
    for question in questions:
        input_dict = {
            "context": page_text,
            "question": question,
        }
        answer = qa_pipeline(input_dict)
        answers.append(answer['answer'])
    return answers

def generate_csv(file_path):
    logger.info(f"Generating CSV file for file: {file_path}")
    qa_pairs = []
    for page_text in file_processing(file_path):
        questions = generate_questions(page_text)
        answers = generate_answers(page_text, questions)
        for i, (question, answer) in enumerate(zip(questions, answers), start=1):
            qa_pairs.append((i, question, answer))
    
    # Create output folder if it doesn't exist
    output_folder = 'static/output'
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, "QA.csv")

    # Write QA pairs to CSV file
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Page", "Question", "Answer"])  # Writing the header row
        csv_writer.writerows(qa_pairs)
    logger.info(f"CSV file generated: {output_file}")

    return output_file

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_file(request: Request, pdf_file: bytes = File(...), filename: str = Form(...)):
    # Save uploaded PDF file
    base_folder = 'static/docs'
    os.makedirs(base_folder, exist_ok=True)
    pdf_filename = os.path.join(base_folder, filename)
    async with aiofiles.open(pdf_filename, 'wb') as f:
        await f.write(pdf_file)
    logger.info(f"File uploaded: {pdf_filename}")
    return JSONResponse(content={"msg": 'success', "pdf_filename": pdf_filename})

@app.post("/analyze")
async def analyze_pdf(request: Request, pdf_filename: str = Form(...)):
    # Analyze PDF file and generate QA pairs
    output_file = generate_csv(pdf_filename)
    return JSONResponse(content={"output_file": output_file})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host='0.0.0.0', port=8000, reload=True)
