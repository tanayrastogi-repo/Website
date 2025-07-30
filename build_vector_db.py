# import dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
import json
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
import logging

# ## Load the environment file
# dotenv.load_dotenv()

# Folder containing PDF files
folder_path = Path("docs")

## Embedding Model from Google Gemini
# Explicitly pass the API key from the environment variable
# GoogleGenerativeAIEmbeddings will look for GOOGLE_API_KEY by default,
# but sometimes explicit passing is cleaner or necessary depending on library version/setup.
# Let's ensure it's robust by passing it directly from os.getenv().
google_api_key = os.getenv("GOOGLE_API_KEY")
print("GOOGLE_API_KEY_CHECK:", google_api_key)
if not google_api_key:
    logging.error("GOOGLE_API_KEY environment variable is not set. Cannot initialize Google Generative AI Embeddings.")
    raise ValueError("GOOGLE_API_KEY is missing.")

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)




##------- Functions -------##
def setup_logging():
    """
    Configures the Python logging library to write to 'vecDataHist.txt'.
    Messages will be appended with a date/time stamp.
    """
    # Define the log file name
    log_file_name = "vecDataHist.txt"

    # Configure the basic logging setup
    logging.basicConfig(
        filename=log_file_name,
        filemode='a',            # 'a' for append mode (creates file if it doesn't exist)
        format='{asctime} - {levelname} - {message}', # Format: <date> <time> - <level> - <message>
        style='{',
        datefmt='%Y-%m-%d %H:%M:%S', # Specific date/time format, 
        level=logging.INFO       # Set the logging level (e.g., INFO, DEBUG, WARNING, ERROR, CRITICAL)
    )
    # Add a StreamHandler for console outputs
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)




def get_current_pdf_files(folder: Path) -> set:
    """Returns a set of PDF file names in the given folder."""
    pdf_files = set()
    for file_path in folder.iterdir():
        if file_path.is_file() and file_path.suffix.lower() == '.pdf':
            pdf_files.add(file_path.name)
    return pdf_files

def load_processed_files_record(record_path: str) -> set:
    """Loads the set of processed file names from a JSON file."""
    if os.path.exists(record_path):
        with open(record_path, 'r') as f:
            return set(json.load(f))
    return set()

def save_processed_files_record(record_path: str, files_set: set):
    """Saves the set of processed file names to a JSON file."""
    with open(record_path, 'w') as f:
        json.dump(list(files_set), f, indent=4)




# Set up the logging configuration
setup_logging()


##------- Load the files to create database -------##
# Ensure folder exists
if not folder_path.exists() or not folder_path.is_dir():
    raise FileNotFoundError(f"The folder '{folder_path}' does not exist or is not a directory.")

# Get current PDF files in the docs folder
current_pdf_files = get_current_pdf_files(folder_path)

# Load previously processed files record
processed_files_on_record = load_processed_files_record(os.getenv('PROCESSED_FILES_RECORD'))


##------- Embedding the files -------##
# Determine if a full re-embedding is needed
# This happens if:
# 1. The Chroma database directory doesn't exist.
# 2. There are new PDF files in 'docs' not present in our record.
# 3. Any files previously processed are now missing (optional, but good for cleanup/rebuild)
needs_rebuild = not os.path.exists(os.getenv('CHROMA_PERSIST_DIRECTORY')) or \
                not current_pdf_files.issubset(processed_files_on_record) or \
                not processed_files_on_record.issubset(current_pdf_files)


if needs_rebuild:
    logging.info("New or updated files detected, or Chroma DB not found. Building/updating vector database...")

    # Variable list for all loaded documents
    docs = []

    # Iterate over each file in the folder
    for file_path in folder_path.iterdir():
        try:
            # Ensure it's a file (not a subfolder)
            if not file_path.is_file():
                continue

            # Check if file has an extension
            if file_path.suffix == '':
                raise logging.error(ValueError(f"File '{file_path.name}' has no extension."))

            # Load for PDF
            if file_path.suffix.lower() == '.pdf':
                loader = PyPDFLoader(file_path)
                loaded_docs = loader.load()
                docs.extend(loaded_docs)
            else:
                logging.info(f"Document type {file_path.name} is not supported yet and will be skipped.")

        except Exception as e:
            logging.error(f"Error processing file '{file_path.name}': {e}")

        finally:
            # Explicitly delete the file_path variable (good practice, though not strictly necessary in this loop)
            del file_path
    logging.info(f"Loaded {len(docs)} pages from all the documents.")



    ##------- Splitting documents -------##
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,      # chunk size (characters)
        chunk_overlap=2000,   # chunk overlap (characters)
        add_start_index=True, # track index in original document
    )
    all_splits = text_splitter.split_documents(docs)
    logging.info(f"Split docs post into {len(all_splits)} sub-documents.")


    ##------- Vector Database -------##
    ## Embedding the documents and creating/updating the vector database
    # Initialize Chroma to persist new data or overwrite existing
    vector_store = Chroma.from_documents(
        documents=all_splits,
        embedding=embeddings,
        collection_name=os.getenv('COLLECTION_NAME'),
        persist_directory=os.getenv('CHROMA_PERSIST_DIRECTORY'),
    )
    logging.info(f"Vector database built/updated and saved to {os.getenv('CHROMA_PERSIST_DIRECTORY')}")

    # Save the current set of PDF files as processed
    save_processed_files_record(os.getenv('PROCESSED_FILES_RECORD'), current_pdf_files)

else:
    logging.info("No new or updated files detected. Loading existing vector database is up-to-date")
