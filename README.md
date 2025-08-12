# Vector Database Automation
This project uses a Python script and a GitHub Actions workflow to automatically build and update a Chroma vector database from PDF files.

## How It Works
A GitHub Action (.github/workflows/build-vector-db.yml) monitors the docs/ directory. When PDF files are added or changed, the workflow executes the build_vector_db.py script.

The script:

 - Scans the docs/ directory for PDF files.
 - Compares the current files against a record of previously processed files (processed_files_record.json).
 - If changes are detected or the database doesn't exist, it rebuilds the database.
 - It extracts text from the PDFs, splits it into chunks, and generates embeddings using Google's models/embedding-001.
 - The embeddings are stored in a persistent Chroma vector database located in chroma_langchain_db/.
 - The workflow then commits the updated database, the processed files record, and a log file (vecDataHist.txt) back to the repository.

## Project Structure
-   `build_vector_db.py`: The main Python script for building the vector database.
    
-   `.github/workflows/build-vector-db.yml`: The GitHub Actions workflow that automates the build process.
    
-   `docs/`: Directory containing the source PDF documents.
    
-   `requirements.txt`: A file listing the required Python packages.
    
-   `chroma_langchain_db/`: The output directory for the persistent Chroma vector database.
    
-   `processed_files_record.json`: A record of files that have been processed to track changes.
    
-   `vecDataHist.txt`: Log file for the database build history.
