from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
from flask import Flask, request, jsonify, render_template
from src.loader import load_pdf
from src.vector_store import get_vector_store
from src.qa_chain import build_qa_chain

load_dotenv()


app = Flask(__name__)

# (other imports stay the same)

@app.route('/')
def index():
    return render_template('index.html')

pdf_path = os.getenv("PDF_PATH", "data/medical_book.pdf")
text = load_pdf(pdf_path)
vectorstore = get_vector_store(text)
qa = build_qa_chain(vectorstore)

@app.route('/ask', methods=['POST'])
def ask():
    question = request.json.get("question")
    answer = qa.run(question)
    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(port=8080, debug=True)
