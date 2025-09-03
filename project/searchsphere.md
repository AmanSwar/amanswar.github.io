
## **Search Sphere - Technical Documentation**

### **1. Introduction**

Search Sphere is a standalone, AI-powered semantic search engine that runs locally on a user's machine. It is designed to overcome the limitations of traditional keyword-based file search by understanding the *meaning* behind a user's query. By leveraging state-of-the-art machine learning models, it can find relevant documents and images even if the query's keywords are not present in the file's name or content.

This document provides a detailed technical overview of the system's architecture, pipelines, and core components.

### **2. System Architecture**

The application is divided into two main packages: `encoder` and `query`, which work together to provide the core functionality. The `run.py` script serves as the main entry point, orchestrating the overall workflow and managing the user interface.

-   **`run.py` (Main Application Runner):**
    -   Handles all user interaction through a CLI built with the `Rich` library.
    -   Initializes the application, including a startup sequence and welcome message.
    -   Prompts the user for a directory to index.
    -   Calls the `encoder` module to perform the indexing.
    -   Calls the `query` module to handle the interactive search loop.

-   **`encoder` Package (The Indexer):**
    -   **`main_seq.py`:** Contains the primary logic for the indexing pipeline. It traverses the file system, orchestrates content extraction and embedding generation, and saves the final index.
    -   **`embedding.py`:** A critical module that uses the **MobileCLIP** model to convert text strings and image files into 512-dimensional vector embeddings.
    -   **`faiss_base.py`:** A wrapper class, `FAISSManagerHNSW`, that abstracts away the complexity of managing the **FAISS** vector indexes. It handles adding new vectors, training the index (if necessary), and saving/loading the index from disk.
    -   **`utils.py`:** A set of helper functions for tasks like extracting text from various file formats (`.pdf`, `.docx`, etc.) and retrieving file metadata.

-   **`query` Package (The Searcher):**
    -   **`query.py`:** The heart of the search functionality. It takes a user's raw query, uses the `utils` module to determine the query's intent, generates a query embedding, and performs the search against the FAISS index.
    -   **`utils.py`:** Contains the `index_token` function, which loads the fine-tuned **MobileBERT** model to classify the user's query as either `TEXT` or `IMAGE`.
    -   **`fine_tune.py`:** A utility script (not part of the main application flow) used to train the MobileBERT model for the intent classification task.

### **3. The Indexing Pipeline**

The indexing process is a sequential flow that builds the knowledge base for the search engine. This is handled by `encoder/main_seq.py`.

1.  **Initialization:** The `FAISSManagerHNSW` is initialized, and any existing FAISS indexes (`index/text_index.index`, `index/image_index.index`) and metadata files are loaded into memory.

2.  **File Traversal:** The application walks through the user-specified directory (`os.walk`) and compiles a list of all files that match the supported extensions (defined in `encoder/config.py`).

3.  **Content Processing Loop:** Each file is processed one by one:
    a.  **Content Extraction:** Based on the file extension, the appropriate function from `encoder/utils.py` is called to extract the content. For text files, this is the raw text; for images, it is the file path.
    b.  **Embedding Generation:** The extracted content (or image path) is passed to the `embedding.py` module. The `MobileCLIP` model then generates a 512-dimension floating-point vector.
    c.  **Temporary Storage:** The generated embedding and its associated metadata (file path, name) are temporarily stored in a list within the `FAISSManagerHNSW` instance.

4.  **Batch Indexing:** Once all files have been processed, the `train_add` method of the `FAISSManagerHNSW` is called. This method stacks all the temporarily stored embeddings into a single NumPy array and adds them to the FAISS index. The `HNSWFlat` index type is highly efficient for this, as it does not require an explicit training step like other FAISS indexes (e.g., IVF).

5.  **Save to Disk:** Finally, the `save_state` method is called to write the updated FAISS indexes and metadata JSON files to the `index/` directory, ensuring persistence between sessions.

### **4. The Search Pipeline**

The search process is designed to be fast and accurate, providing relevant results in real-time. This is handled by `query/query.py`.

1.  **User Input:** The user enters a natural language query into the CLI.

2.  **Query Intent Classification:**
    a.  The query string is passed to the `index_token` function in `query/utils.py`.
    b.  This function uses a fine-tuned `MobileBERT` model to perform a sequence classification task. The model outputs a label, either `TEXT` or `IMAGE`, based on what it believes the user is looking for.

3.  **Query Embedding:**
    a.  The same query string is passed to the `text_extract` function in `encoder/embedding.py`.
    b.  The `MobileCLIP` model converts the query into a 512-dimensional vector embedding. This is the same model used for indexing, which is crucial for ensuring that the query and the indexed content exist in the same "embedding space."

4.  **Similarity Search:**
    a.  Based on the intent (`TEXT` or `IMAGE`), the appropriate search method (`search_text` or `search_image`) in `FAISSManagerHNSW` is called.
    b.  FAISS then performs a k-Nearest Neighbor (k-NN) search on the corresponding vector index. It calculates the distance between the query vector and all the vectors in the index and returns the `k` vectors with the smallest distance (i.e., the highest similarity).

5.  **Results Presentation:**
    a.  The search returns the IDs and similarity scores of the top matching items.
    b.  The application looks up the metadata (file name and path) for these IDs from the loaded metadata map.
    c.  The final results are displayed to the user in a formatted table using the `Rich` library.

### **5. Key Technologies and Rationale**

-   **MobileCLIP:** Chosen for its excellent balance of performance and efficiency. It is designed for mobile and edge devices, making it lightweight enough to run quickly on a local machine while still providing high-quality, cross-modal embeddings for both text and images.

-   **FAISS (HNSWFlat):** The `HNSW (Hierarchical Navigable Small World)` graph-based index was chosen for its exceptional search speed and accuracy, especially for large datasets. Unlike `IVF` indexes, `HNSW` does not require a separate training phase, which makes it ideal for this application where new files can be added dynamically.

-   **MobileBERT:** A compact and fast variant of the BERT model. It was chosen for the query classification task because it is lightweight and provides excellent performance on classification tasks without introducing significant latency into the search pipeline.

-   **Rich:** This library was chosen to create a modern, visually appealing, and user-friendly CLI. It provides out-of-the-box components for progress bars, formatted tables, and styled text, which greatly enhances the overall user experience compared to a standard print-based interface.
