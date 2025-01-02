import csv
import os
import glob
import re
import unicodedata
import time
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template, send_file

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

# Global variable to track pipeline status
pipeline_status = {"status": "Idle", "progress": 0}

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

# Route: Run Pipeline
@app.route("/run", methods=["POST"])
def run_pipeline():
    global pipeline_status

    # Extract folder path and output file from request
    folder_path = request.form.get("folder_path", "./substack_articles")
    output_csv = request.form.get("output_csv", "output_tweets.csv")

    print(f"Starting pipeline with folder_path: {folder_path}, output_csv: {output_csv}", flush=True)

    # Validate folder path
    if not os.path.exists(folder_path):
        pipeline_status = {"status": f"Folder '{folder_path}' does not exist.", "progress": 0}
        print(f"Error: {pipeline_status['status']}", flush=True)
        return jsonify({"error": pipeline_status["status"]}), 400

    # Load articles and filenames
    start_time = time.time()
    file_names, articles = load_substack_articles(folder_path)
    print(f"Loaded {len(articles)} articles in {time.time() - start_time:.2f} seconds.", flush=True)

    if not articles:
        pipeline_status = {"status": "No articles found in the specified folder.", "progress": 0}
        print(f"Error: {pipeline_status['status']}", flush=True)
        return jsonify({"error": pipeline_status["status"]}), 400

    final_results = []
    pipeline_status = {"status": "Running", "progress": 0}

    # Process each article
    for idx, (name, article) in enumerate(zip(file_names, articles)):
        print(f"Processing Article {idx+1}: {name}", flush=True)
        pipeline_status["progress"] = int(((idx + 1) / len(articles)) * 100)
        pipeline_status["status"] = f"Processing Article {idx+1} of {len(articles)}: {name}"

        try:
            start_time = time.time()
            analyze_task = Task(
                name=f"Analyze Article {idx+1}",
                agent=analyzer_agent,
                description=f"Summarize the following article and extract tweetable points:\n\n{article}",
                expected_output="A concise summary and key themes for tweet generation.",
                timeout=60
            )

            generate_tweet_task = Task(
                name=f"Generate Tweets for Article {idx+1}",
                agent=tweet_generator_agent,
                description=(
                    "Based on the article summary, generate a list of 5 engaging tweets. "
                    "Each tweet must be at least 2 sentences long and avoid using any hashtags (#). "
                    "Do not number the tweets or include bullet points. Output each tweet as plain text."
                ),
                expected_output="A plain text list of 5 well-crafted tweets, each at least 2 sentences long.",
                timeout=60
            )

            review_task = Task(
                name=f"Review Tweets for Article {idx+1}",
                agent=reviewer_agent,
                description=(
                    "Refine the generated tweets for clarity and engagement. "
                    "Each tweet must be at least 2 sentences long and avoid using any hashtags (#)."
                ),
                expected_output="A refined and polished list of tweets.",
                timeout=60
            )

            # Run the tasks sequentially
            task_start_time = time.time()
            print(f"Starting Crew for Article {idx+1}", flush=True)
            crew = Crew(
                agents=[analyzer_agent, tweet_generator_agent, reviewer_agent],
                tasks=[analyze_task, generate_tweet_task, review_task],
                verbose=True
            )

            result = crew.kickoff()
            print(f"Crew completed in {time.time() - task_start_time:.2f} seconds.", flush=True)

            # Extract and clean results
            result_start_time = time.time()
            output_text = result.content if hasattr(result, 'content') else str(result)
            cleaned_tweets = [clean_text(tweet) for tweet in output_text.strip().split("\n") if tweet.strip()]
            final_results.append((name, cleaned_tweets))
            print(f"Result processing took {time.time() - result_start_time:.2f} seconds.", flush=True)

        except Exception as e:
            print(f"Error processing Article {idx+1}: {e}", flush=True)
            pipeline_status["status"] = f"Error processing Article {idx+1}: {e}"
            return jsonify({"error": str(e)}), 500

    # Save results to CSV
    print("Saving results to CSV.", flush=True)
    csv_start_time = time.time()
    with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Original Article Name", "Final Tweet"])
        for name, tweets in final_results:
            for tweet in tweets:
                writer.writerow([name, tweet])
    print(f"CSV saved in {time.time() - csv_start_time:.2f} seconds.", flush=True)

    pipeline_status = {"status": "Completed", "progress": 100}
    print("Pipeline completed successfully.", flush=True)
    return jsonify({"message": "Pipeline completed successfully.", "output_csv": output_csv})

# Route: Get Pipeline Status
@app.route("/status", methods=["GET"])
def get_status():
    return jsonify(pipeline_status)

# Route: Download CSV
@app.route("/download", methods=["GET"])
def download_csv():
    file_path = "output_tweets.csv"
    print(f"Download request received for file: {file_path}", flush=True)

    if not os.path.exists(file_path):
        print("Error: Output file not found.", flush=True)
        return jsonify({"error": "Output file not found. Please run the pipeline first."}), 400

    print("Sending file to client.", flush=True)
    return send_file(file_path, as_attachment=True)

if __name__ == "__main__":
    print("Starting Flask app...", flush=True)
    try:
        app.run(host="0.0.0.0", port=8000)
    finally:
        print("Flask app has stopped running.", flush=True)






