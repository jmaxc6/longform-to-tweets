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
from supabase import create_client
import uuid  # For generating unique session IDs
import io
import requests
from datetime import datetime, timezone
import shutil

# Initialize Flask app
app = Flask(__name__)

# Initialize Supabase client with error handling
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

if not supabase_url or not supabase_key:
    raise ValueError("Supabase URL or Key is missing. Please check your .env file.")

supabase = create_client(supabase_url, supabase_key)

if supabase is None:
    raise RuntimeError("Failed to initialize Supabase client.")

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

def log_error(message):
    print(f"[ERROR] {message}", flush=True)    

def update_pipeline_status(session_id, status=None, progress=None, details=None, error_message=None, output_file_url=None, reset_at=None, archived=None):
    update_data = {}
    if status is not None:
        update_data["status"] = status
    if progress is not None:
        update_data["progress"] = progress
    if details is not None:
        update_data["details"] = details
    if error_message is not None:
        update_data["error_message"] = error_message
    if output_file_url is not None:
        update_data["output_file_url"] = output_file_url
    if reset_at is not None:
        update_data["reset_at"] = reset_at
    if archived is not None:
        update_data["archived"] = archived

    supabase.table("pipeline_status").update(update_data).eq("session_id", session_id).execute()

def upload_input_to_supabase(file, filename):
    """
    Upload the input file to Supabase storage bucket 'Input Folder'.

    Args:
        file: File object to be uploaded.
        filename: The name of the file to be stored in Supabase.

    Returns:
        Signed URL of the uploaded file.
    """
    try:
        # Upload the file to Supabase storage bucket 'Input Folder'
        response = supabase.storage.from_("Input Folder").upload(filename, file)

        if not response or hasattr(response, "error"):
            raise RuntimeError(f"File upload failed: {getattr(response, 'error', 'Unknown error')}")

        # Generate a signed URL for the uploaded file
        bucket_name = "Input Folder"
        files_in_bucket = supabase.storage.from_(bucket_name).list("")
        print(f"Files in Input Folder bucket: {files_in_bucket}", flush=True)
        print(f"The Filename is {filename}", flush=True)
        print(f"Bucket name: '{bucket_name}'", flush=True)  # This will show if there is whitespace
        
        signed_url_response = supabase.storage.from_(bucket_name).create_signed_url(filename, expires_in=3600)

        if not signed_url_response or not signed_url_response.get("signed_url"):
            raise RuntimeError("Signed URL generation failed or returned None.")

        return signed_url_response["signed_url"]

    except Exception as e:
        print(f"Error uploading input file to Supabase: {str(e)}", flush=True)
        raise RuntimeError(f"Failed to upload input file to Supabase: {str(e)}")

def fetch_pipeline_status(session_id):
    response = supabase.table("pipeline_status").select("*").eq("session_id", session_id).eq("archived", False).execute()
    return response.data[0] if response.data else None

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

def create_pipeline_session(session_id):
    """
    Inserts a new session into the database using the given session_id.
    """
    created_at = datetime.now(timezone.utc).isoformat()  # Get current UTC time
    supabase.table("pipeline_status").insert({
        "session_id": session_id,
        "status": "Running",
        "progress": 0,
        "details": [],
        "error_message": None,
        "output_file_url": None,
        "created_at": created_at,
        "archived": False
    }).execute()

# Route: Home - Serve the GUI
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

# Route: Upload ZIP file and start processing in background
@app.route("/upload", methods=["POST"])
def upload_folder():
    # Check for file in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part provided in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file for upload'}), 400

    if file and allowed_file(file.filename):
        # Generate a unique session ID
        session_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)

        try:
            # Read file content as bytes
            file.stream.seek(0)  # Ensure the file pointer is at the beginning
            file_content = file.stream.read()  # Read the file as bytes

            # Debugging: Print the type and first 100 bytes of file_content
            print(type(file_content), file_content[:100])  # Debugging line

            # Validate that the file content is in bytes format
            if not isinstance(file_content, bytes):
                raise ValueError("File content is not in bytes format.")

            # Upload file to Supabase storage
            response = supabase.storage.from_("Input Folder").upload(
                f"{session_id}/{filename}", file_content
            )

            # Check if the upload failed
            if hasattr(response, "error") and response.error:
                raise RuntimeError(f"Upload failed: {response.error['message']}")
            
            bucket_name = "Input Folder"
            
            print(f"session_id: {session_id}, filename: {filename}", flush = True)
            print(f"Full path: {session_id}/{filename}", flush = True)
            files_in_bucket = supabase.storage.from_(bucket_name).list("")
            print(f"Bucket name: '{bucket_name}'", flush=True)  # Check for unintended spaces
            files_in_bucket = supabase.storage.from_("Input Folder").list(f"{session_id}")
            print(f"Files in {bucket_name}/{session_id}: {files_in_bucket}", flush=True)

            object_path = f"{session_id}/{filename}"

            # signed_url_response = supabase.storage.from_("Input Folder").create_signed_url(
            # "874929bf-d9ec-4679-b600-5b62ebd91a14/Archive-15.zip", expires_in=3600)

            # Generate a signed URL for the uploaded file
            signed_url_response = supabase.storage.from_(bucket_name).create_signed_url(
                object_path, expires_in=3600)

            print(f"The signed URL response is {signed_url_response}", flush=True)
            
            if not signed_url_response or not signed_url_response.get("signedURL"):
                print(f"Signed URL response: {signed_url_response}")
                raise RuntimeError("Signed URL generation failed or returned None.")

            # Create a new pipeline session in the database
            create_pipeline_session(session_id)

            # Start processing in a background thread
            thread = Thread(target=process_file_from_supabase, args=(session_id, signed_url_response["signedURL"]))
            thread.start()

            # Return the session ID and success message
            return jsonify({
                "message": "File uploaded successfully. Processing started.",
                "session_id": session_id,
                "file_url": signed_url_response["signedURL"]
            }), 200
        except Exception as e:
            return jsonify({"error": f"Error uploading input file to Supabase: {str(e)}"}), 500

    return jsonify({'error': 'Invalid file format, only ZIP files are allowed'}), 400

def process_file_from_supabase(session_id, input_file_url):
    try:
        # Fetch the ZIP file from Supabase using the signed URL
        response = requests.get(input_file_url)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch file from Supabase: {response.text}")

        # Extract the ZIP file contents
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            zip_ref.extractall("temporary_extracted_files")  # Extract to a temp folder

        # Process the extracted files
        folder_path = "temporary_extracted_files"
        run_pipeline(folder_path, session_id)
    except Exception as e:
        update_pipeline_status(session_id, status="Error", error_message=str(e))
    finally:
        # Ensure cleanup happens even if an error occurs
        if os.path.exists("temporary_extracted_files"):
            shutil.rmtree("temporary_extracted_files")

def run_pipeline(folder_path, session_id):
    output_csv = f"{session_id}.csv"  # Unique file name based on session_id
    print(f"Starting pipeline with folder_path: {folder_path}, output_csv: {output_csv}", flush=True)

    # Validate folder path
    if not os.path.exists(folder_path):
        update_pipeline_status(session_id, status="Error", error_message=f"Folder '{folder_path}' does not exist.")
        return

    # Load articles and filenames
    file_names, articles = load_substack_articles(folder_path)
    if not articles:
        update_pipeline_status(session_id, status="Error", error_message="No articles found in the specified folder.")
        return

    update_pipeline_status(session_id, status="Running", progress=0, details=[])

    final_results = []

    for idx, (name, article) in enumerate(zip(file_names, articles)):
        progress = int(((idx + 1) / len(articles)) * 100)
        update_pipeline_status(
            session_id,
            progress=progress,
            status=f"Processing Article {idx + 1} of {len(articles)}",
            details=[f"Article {idx + 1} Completed"]
        )

        try:
            # Run tasks sequentially
            analyze_task = Task(name=f"Analyze {name}", agent=analyzer_agent, description=f"Summarize the following article and extract tweetable points:\n\n{article}", expected_output="A concise summary and key themes for tweet generation.", timeout=60)
            generate_tweet_task = Task(name=f"Generate Tweets for {name}", agent=tweet_generator_agent, description=("Based on the article summary, generate a list of 5 engaging tweets. "
                    "Each tweet must be at least 2 sentences long and avoid using any hashtags (#). "
                    "Do not number the tweets or include bullet points. Output each tweet as plain text."), expected_output="A plain text list of 5 well-crafted tweets, each at least 2 sentences long.", timeout=60)
            review_task = Task(name=f"Review Tweets for {name}", agent=reviewer_agent, description=("Refine the generated tweets for clarity and engagement. "
                    "Each tweet must be at least 2 sentences long and avoid using any hashtags (#)."), expected_output="A refined and polished list of tweets.", timeout=60)

            crew = Crew(agents=[analyzer_agent, tweet_generator_agent, reviewer_agent], tasks=[analyze_task, generate_tweet_task, review_task])
            result = crew.kickoff()

            output_text = result.content if hasattr(result, 'content') else str(result)
            cleaned_tweets = [clean_text(tweet) for tweet in output_text.split("\n") if tweet.strip()]
            final_results.append((name, cleaned_tweets))
            print("Successfully processed article:", name)
        except Exception as e:
            update_pipeline_status(session_id, status="Error", error_message=f"Error processing article {name}: {str(e)}")
            return

    # Save results to CSV
    try:
        print("Writing results to CSV...")
        with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Original Article Name", "Final Tweet"])
            for name, tweets in final_results:
                for tweet in tweets:
                    writer.writerow([name, tweet])
        print("CSV file written successfully:", output_csv)

        # Upload the CSV to Supabase storage
        try:
            with open(output_csv, "rb") as file:
                response = supabase.storage.from_("Zip uploads").upload(output_csv, file)
            
            print(f"Upload response: {response}")  # Debugging output

            # Check for upload error
            if isinstance(response, dict) and "error" in response:
                raise Exception(f"Error uploading file to Supabase: {response['error']}")

            # Generate a signed URL
            signed_url_response = supabase.storage.from_("Zip uploads").create_signed_url(output_csv, expires_in=3600)
            print(f"Signed URL response: {signed_url_response}")  # Debugging output

            # Check for signed URL error
            if isinstance(signed_url_response, dict) and "error" in signed_url_response:
                raise Exception(f"Error generating signed URL: {signed_url_response['error']}")

            signed_url = signed_url_response.get("signedURL", None)
            if not signed_url:
                raise Exception("Signed URL generation failed or returned None.")

            print("Signed URL generated successfully:", signed_url)

            update_pipeline_status(
                session_id,
                status="Pipeline Finished",
                progress=100,
                details=["Pipeline Finished"],
                output_file_url=signed_url
            )
            print("Pipeline status updated successfully with signed URL.")
        except Exception as e:
            update_pipeline_status(session_id, status="Error", error_message=f"Failed to upload results: {str(e)}")
    except Exception as e:
        update_pipeline_status(session_id, status="Error", error_message=f"Failed to save results: {str(e)}")
    finally:
        # Clean up the local file after upload
        try:
            if os.path.exists(output_csv):
                os.remove(output_csv)
                print(f"Temporary file {output_csv} removed successfully.")
        except Exception as e:
            log_error(f"Failed to clean up temporary file {output_csv}: {str(e)}")

# Route: Progress Endpoint
@app.route("/progress", methods=["GET"])
def get_progress():
    session_id = request.args.get("session_id")
    if not session_id:
        return jsonify({"error": "Session ID is required"}), 400

    status = fetch_pipeline_status(session_id)
    if not status:
        return jsonify({"error": "Session not found"}), 404

    return jsonify(status)

@app.route("/download", methods=["GET"])
def download_csv():
    session_id = request.args.get("session_id")
    if not session_id:
        return jsonify({"error": "Session ID is required"}), 400

    # Fetch the pipeline status to get the signed URL
    status = fetch_pipeline_status(session_id)
    if not status:
        return jsonify({"error": "Session not found"}), 404

    signed_url = status.get("output_file_url")
    if not signed_url:
        return jsonify({"error": "Output file not available. The pipeline might still be running or failed."}), 400

    try:
        # Fetch the file from the signed URL
        response = requests.get(signed_url)
        if response.status_code != 200:
            return jsonify({"error": "Failed to fetch the file from the signed URL."}), 400

        # Serve the CSV file directly
        return send_file(
            io.BytesIO(response.content),
            mimetype="text/csv",
            as_attachment=True,
            download_name=f"{session_id}.csv",
        )
    except Exception as e:
        return jsonify({"error": f"Failed to process download: {str(e)}"}), 500

# Route: Reset Endpoint

@app.route("/reset", methods=["POST"])
def reset_pipeline():
    try:
        # Archive all non-archived sessions
        supabase.table("pipeline_status").update({"archived": True}).neq("archived", True).execute()

        # Clean up uploaded files from Supabase
        bucket_name = "Input Folder"
        storage_response = supabase.storage.from_(bucket_name).list("")
        if storage_response and isinstance(storage_response, list):
            for file in storage_response:
                try:
                    supabase.storage.from_(bucket_name).remove([file["name"]])
                    print(f"Deleted file: {file['name']}", flush=True)
                except Exception as e:
                    print(f"Failed to delete file: {file['name']}, Error: {str(e)}", flush=True)

        # Clean up temporary local files/directories
        temp_dir = "temporary_extracted_files"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"Deleted temporary directory: {temp_dir}", flush=True)

        print("Pipeline reset successfully.")
        return jsonify({"message": "Pipeline has been reset successfully. Ready for a new upload."}), 200
    except Exception as e:
        print(f"Error resetting pipeline: {str(e)}", flush=True)
        return jsonify({"error": "An error occurred while resetting the pipeline. Please try again."}), 500

if __name__ == "__main__":
    print("Starting Flask app...", flush=True)
    app.run(host="0.0.0.0", port=8000)



