import json
import logging
import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import worker # Import the worker module

# Initialize Flask app and CORS
app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
app.logger.setLevel(logging.DEBUG)
#app.logger.setLevel(logging.ERROR)

# Define the route for querying the LLM to extract from PDF in vector store

@app.route('/process-purchase-order', methods=['POST'])
def process_purchase_order():
    if request.is_json:
        prompt = f"""
        Populate values from the PDF into the provided JSON formatted template.  Populate only blank JSON fields. Return the JSON as a single line. For any values expressed as metric tons (MT), convert these values into kilograms (KG) by multiplying by 1000. The JSON template is {json.dumps(request.json)}
        """
        app.logger.debug(f"Prompt is {prompt}")
        bot_response = worker.process_prompt(prompt)  # Process the user's message using the worker module
        #bot_response = "OK"
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
    worker.process_document(file_path)  # Process the document using the worker module
    # Return a success message as JSON
    return "OK",200

# Run the Flask app

if __name__ == "__main__":
    app.run(debug=True, port=8000, host='0.0.0.0')
