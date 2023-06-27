from fastapi import FastAPI, File, UploadFile
from starlette.responses import JSONResponse
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pdfGPT

app = FastAPI()

origins = [
    "*",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Online"}


@app.post("/analyze")
async def upload_pdf(pdf: UploadFile = File(...)):
    try:
        # Save the uploaded file
        saved_file_path = f"uploaded_files/{pdf.filename}"
        with open(saved_file_path, "wb") as file:
            file.write(await pdf.read())

        openAI_key="placeholder"
        url = ""
        question = "debug" # TODO
        answer = pdfGPT.question_answer(url, saved_file_path, question, True, "intermediate", openAI_key)

        return JSONResponse(content={"message": answer}, status_code=200)
    except Exception as e:
        print(f"Error uploading file: {e}")
        return JSONResponse(content={"message": "Error uploading file"}, status_code=500)