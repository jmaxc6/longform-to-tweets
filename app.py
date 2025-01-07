import csv
import os
import glob
import re
import unicodedata
import time
import zipfile
from threading import Thread, Lock
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Verify API key
if not openai_api_key:
    raise ValueError("OpenAI API Key not found. Please check your .env file.")

# Initialize OpenAI GPT-4 model
chat_model = ChatOpenAI(model="gpt-4", temperature=0.7, openai_api_key=openai_api_key)

# Define Agents
analyzer_agent = Agent(
    role="Content Analyzer",
    goal="Summarize and extract key tweetable points from Substack articles.",
    backstory="You are a content strategist who identifies engaging themes and key points.",
    verbose=True
)

tweet_generator_agent = Agent(
    role="Tweet Generator",
    goal="Generate engaging tweets from summarized content.",
    backstory="You are an expert social media copywriter, skilled in writing viral tweets.",
    verbose=True
)

reviewer_agent = Agent(
    role="Tweet Reviewer",
    goal="Review and refine generated tweets for clarity and engagement.",
    backstory="You ensure tweets are polished, engaging, and aligned with the author's voice.",
    verbose=True
)

# Global variables
pipeline_status = {"status": "Idle", "progress": 0, "details": []}
pipeline_lock = Lock()  # To make `pipeline_status` updates thread-safe

# Set upload folder and allowed extensions
UPLOAD_FOLDER = "uploaded_files"
ALLOWED_EXTENSIONS = {'zip'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Utility function to load Substack articles
def load_substack_articles(folder_path):
    articles = []
    file_names = []
    for file_path in glob.glob(f"{folder_path}/*.txt"):
        with open(file_path, 'r', encoding='utf-8') as file:
            articles.append(file.read())
            file_names.append(os.path.basename(file_path))  # Track article names
    return file_names, articles

# Utility function to clean text
def clean_text(text):
    clean = unicodedata.normalize("NFKD", text)  # Normalize the text
    clean = re.sub(r'[^\x00-\x7F]+', ' ', clean)  # Replace non-ASCII characters
    clean = re.sub(r"^\s*\d+[\.\)]\s*|\s*[\u2022\u25CF]\s*", "", clean)  # Remove bullet points and numbers
    clean = re.sub(r"\s+", " ", clean)  # Collapse multiple spaces into one
    return clean.strip()  # Remove leading and trailing spaces

# Route: Home - Serve the GUI
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

# Route: Upload ZIP file and start processing in background
@app.route("/upload", methods=["POST"])
def upload_folder():
    global pipeline_status

    if 'file' not in request.files:
        return jsonify({'error': 'No file part provided in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file for upload'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Set pipeline status to running immediately
        with pipeline_lock:
            pipeline_status = {"status": "Running", "progress": 0, "details": []}
        print(f"Pipeline status set to running: {pipeline_status}")

        # Start processing in a background thread
        thread = Thread(target=process_file, args=(filepath,))
        thread.start()

        # Respond to the client immediately
        return jsonify({"message": "File uploaded successfully. Processing started."}), 200

    return jsonify({'error': 'Invalid file format, only ZIP files are allowed'}), 400

def process_file(filepath):
    """
    Extracts and processes the uploaded file in the background.
    """
    extract_path = os.path.join(app.config['UPLOAD_FOLDER'], "extracted")
    os.makedirs(extract_path, exist_ok=True)

    try:
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
    except zipfile.BadZipFile:
        with pipeline_lock:
            pipeline_status["status"] = "Error: Invalid ZIP file"
        return

    # Process the extracted files
    folder_path = extract_path
    run_pipeline(folder_path)

# Pipeline logic
def run_pipeline(folder_path):
    global pipeline_status

    output_csv = "output_tweets.csv"
    print(f"Starting pipeline with folder_path: {folder_path}, output_csv: {output_csv}", flush=True)

    # Validate folder path
    if not os.path.exists(folder_path):
        with pipeline_lock:
            pipeline_status = {"status": f"Folder '{folder_path}' does not exist.", "progress": 0}
        return

    # Load articles and filenames
    file_names, articles = load_substack_articles(folder_path)
    if not articles:
        with pipeline_lock:
            pipeline_status = {"status": "No articles found in the specified folder.", "progress": 0}
        return

    with pipeline_lock:
        pipeline_status = {"status": "Running", "progress": 0, "details": []}

    final_results = []

    for idx, (name, article) in enumerate(zip(file_names, articles)):
        with pipeline_lock:
            pipeline_status["progress"] = int(((idx + 1) / len(articles)) * 100)
            pipeline_status["status"] = f"Article {idx + 1} Completed"
            pipeline_status["details"].append(f"Article {idx + 1} Completed")
        print(f"Updated pipeline status: {pipeline_status}")

        try:
            # Define and execute tasks
            analyze_task = Task(
                name=f"Analyze {name}",
                agent=analyzer_agent,
                description=f"Summarize and analyze the following article:\n\n{article}",
                expected_output="A summary and key themes for tweet generation.",
                timeout=60
            )
            generate_tweet_task = Task(
                name=f"Generate Tweets for {name}",
                agent=tweet_generator_agent,
                description="Generate 5 engaging tweets based on the analysis. Avoid using quotes at the beginning or end and do not include hashtags.",
                expected_output="5 engaging and creative tweets without quotes or hashtags.",
                timeout=60
            )
            review_task = Task(
                name=f"Review Tweets for {name}",
                agent=reviewer_agent,
                description="Refine tweets for clarity and engagement. Ensure the tweets do not use quotes at the start or end and contain no hashtags.",
                expected_output="Polished and refined tweets without quotes or hashtags.",
                timeout=60
            )

            # Run tasks sequentially
            crew = Crew(
                agents=[analyzer_agent, tweet_generator_agent, reviewer_agent],
                tasks=[analyze_task, generate_tweet_task, review_task]
            )
            result = crew.kickoff()

            # Extract results
            output_text = result.content if hasattr(result, 'content') else str(result)
            cleaned_tweets = [clean_text(tweet) for tweet in output_text.split("\n") if tweet.strip()]
            final_results.append((name, cleaned_tweets))

        except Exception as e:
            with pipeline_lock:
                pipeline_status = {"status": f"Error processing {name}: {str(e)}"}
            return

    # Save results to CSV
    with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Original Article Name", "Final Tweet"])
        for name, tweets in final_results:
            for tweet in tweets:
                writer.writerow([name, tweet])

    # Update pipeline status to finished
    with pipeline_lock:
        pipeline_status = {"status": "Pipeline Finished", "progress": 100, "details": ["Pipeline Finished"]}
    print(f"Pipeline finished: {pipeline_status}")

# Route: Progress Endpoint
@app.route("/progress", methods=["GET"])
def get_progress():
    global pipeline_status

    # Log the current progress for debugging purposes
    print(f"Pipeline progress requested: {pipeline_status}")

    with pipeline_lock:
        return jsonify(pipeline_status)

# Route: Reset Endpoint
@app.route("/reset", methods=["POST"])
def reset_pipeline():
    global pipeline_status

    with pipeline_lock:
        pipeline_status = {"status": "Idle", "progress": 0, "details": []}
    print("Pipeline reset to initial state.")

    return jsonify({"message": "Pipeline has been reset."}), 200

# Route: Download CSV
@app.route("/download", methods=["GET"])
def download_csv():
    file_path = "output_tweets.csv"
    if not os.path.exists(file_path):
        return jsonify({"error": "Output file not found. Please run the pipeline first."}), 400
    return send_file(file_path, as_attachment=True)

if __name__ == "__main__":
    print("Starting Flask app...", flush=True)
    app.run(host="0.0.0.0", port=8000)



