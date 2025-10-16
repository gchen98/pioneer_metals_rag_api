import json
import logging
import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import hf_worker as worker # Import the worker module

# Initialize Flask app and CORS
app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
app.logger.setLevel(logging.DEBUG)
#app.logger.setLevel(logging.ERROR)

# Define the route for querying the LLM to extract from PDF in vector store

@app.route('/process-purchase-order', methods=['POST'])
def process_purchase_order():
    if request.is_json:
        #prompt_test = "What are the values of the booking number and containers?"
        #app.logger.debug(f"Prompt is: {prompt_test}")
        #bot_response = "OK"
        prompt_json = json.dumps(request.json)
        bot_response = worker.get_response(prompt_json)
        return bot_response,200
    else:
        return "Not a JSON POST request",400

# Define the route for storing the PDF for RAG

@app.route('/process-pdf', methods=['POST'])
def process_pdf_route():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return "You need to upload a PDF",400

    file = request.files['file']  # Extract the uploaded file from the request
    file_path = file.filename  # Define the path where the file will be saved
    file.save(file_path)  # Save the file
    worker.load_document(file_path)  # Process the document using the worker module
    # Return a success message as JSON
    return "OK",200

# Run the Flask app

if __name__ == "__main__":
    app.run(debug=True, port=8000, host='0.0.0.0')
