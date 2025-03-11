# STORAGE_FILE="scraped_content.txt"
# def scrape_and_store(url, base_url, visited=set()):
#     """Scrapes content from a webpage and explores only internal links recursively."""
#     try:
#         if url in visited:
#             return "Already visited. Skipping."
        
#         visited.add(url)
#         headers = {"User-Agent": "Mozilla/5.0"}
#         response = requests.get(url, headers=headers, timeout=10)
#         if response.status_code != 200:
#             return ""
        
#         soup = BeautifulSoup(response.text, "html.parser")
#         for tag in soup(["nav", "footer", "script", "style", "aside", "noscript"]):
#             tag.decompose()
        
#         extracted_text = []
#         for tag in soup.find_all(["h1", "h2", "h3","h4","h5","h6", "p"]):
#             text = tag.get_text(" ", strip=True)
#             if text:
#                 extracted_text.append(text)
        
#         content = "\n".join(extracted_text)
#         with open(STORAGE_FILE, "a", encoding="utf-8") as file:
#             file.write(f"\n\n--- Content from {url} ---\n\n{content}")
        
#         # Extract and follow only internal links
#         for link in soup.find_all("a", href=True):
#             next_link = link["href"]
#             if next_link.startswith("/"):
#                 next_link = base_url + next_link
#             if next_link.startswith(base_url) and next_link not in visited:
#                 scrape_and_store(next_link, base_url, visited)
        
#         return "Content saved successfully."
#     except requests.exceptions.RequestException:
#         return "Error fetching the page."

# # Example usage
# url = "https://toddlersapp.com/documentation"  # Replace with your target website
# base_url = "https://toddlersapp.com"  # Replace with your target website

# scrape_and_store(url,base_url)




#SEARCHER 
# def search_faiss(query, top_k=3):
#     """Flask API endpoint for FAISS search"""
    
#     # ✅ Check if query is empty and ask for input
#     if not query.strip():
#         return "Please provide a query to search."
    
#     global first_query  # Track if it's the first query

#     # ✅ Extract greeting, appreciation, and actual query
#     greeting, question = extract_greeting_and_question(query)
    
#     # ✅ Debugging print to check extracted values
#     print(f"DEBUG - Greeting: {greeting}, Question: {question}")
    
#     # ✅ If it's an appreciation message, return a polite response
#     if greeting == "Appreciation":
#         return random.choice(APPRECIATION_RESPONSES)
    
#     if greeting == "Help":
#         return random.choice(HELP_RESPONSE)

#     # ✅ Handle greetings: If only a greeting exists, return a greeting response
#     if greeting and question is None:
#         return get_greeting_response(greeting)

#     # ✅ If both a greeting AND a question exist, return both
#     greeting_text = get_greeting_response(greeting) if greeting and not question else ""

    
#     # ✅ If no question is found, return a generic response
#     if question is None:
#         return f"{greeting_text}\n\nHow can I assist you today?" if greeting_text else "How can I assist you today?"

#     # ✅ Normalize the query to handle variations
#     normalized_question = normalize_query(question)
#     print(f"DEBUG - Normalized Question: {normalized_question}")

#     # ✅ Ensure FAISS index exists
#     if not os.path.exists(FAISS_INDEX_FILE) or not os.path.exists(EMBEDDINGS_FILE):
#         return "No FAISS index found. Please build the index first."

#     try:
#         index = faiss.read_index(FAISS_INDEX_FILE)
#     except Exception as e:
#         return f"Error loading FAISS index: {e}"

#     try:
#         content = np.load(EMBEDDINGS_FILE, allow_pickle=True)
#     except Exception as e:
#         return f"Error loading stored text: {e}"

#     # Convert query into an embedding
#     query_embedding = model.encode([normalized_question], convert_to_numpy=True, normalize_embeddings=True)

#     # Perform FAISS search
#     distances, indices = index.search(query_embedding, top_k)

#     # ✅ Check if FAISS returned valid results
#     if indices is None or len(indices[0]) == 0:
#         return "No relevant results found"

#     results = [content[i] for i in indices[0] if 0 <= i < len(content)]

#     # ✅ Remove duplicates & short responses
#     results = list(dict.fromkeys(results))  # Remove duplicates
#     results = [r for r in results if len(r.split()) > 10]  # Ensure meaningful text

#     if not results:
#         return "No relevant results found."

#     # ✅ Generate summary
#     summarized_response = summarize_text(" ".join(results), query=normalized_question, include_greeting=(greeting is not None))

#     # ✅ Adjust response wording based on query
#     final_response = adjust_response_wording(summarized_response, query)

#     # ✅ Combine greeting & final response if greeting exists
#     return f"{greeting_text}\n\n{final_response}" if greeting_text else final_response

#SEARCHER2
# def search_faiss(query, top_k=3):
#     """Searches FAISS index for relevant answers using best similarity metric."""
    
#     # Extract greeting, appreciation, and actual question
#     greeting, question = extract_greeting_and_question(query)
#     print(f"DEBUG - Greeting: {greeting}, Question: {question}")

#     # Handle appreciation messages
#     if greeting == "Appreciation":
#         return random.choice(APPRECIATION_RESPONSES)
    
#     if greeting == "Help":
#         return random.choice(HELP_RESPONSE)

#     # Handle greetings (if only a greeting exists)
#     if greeting and question is None:
#         return get_greeting_response(greeting)

#     # If no question found, return a generic response
#     if question is None:
#         return f"{get_greeting_response(greeting)}\n\nHow can I assist you today?" if greeting else "How can I assist you today?"

#     # Normalize query
#     normalized_question = normalize_query(question)
#     print(f"DEBUG - Normalized Question: {normalized_question}")

#     # Check if FAISS index exists
#     if not os.path.exists(FAISS_INDEX_FILE) or not os.path.exists(EMBEDDINGS_FILE):
#         return "No FAISS index found. Please build the index first."

#     # Load FAISS index
#     try:
#         index = faiss.read_index(FAISS_INDEX_FILE)
#     except Exception as e:
#         return f"Error loading FAISS index: {e}"

#     # Load stored chunks (instead of full paragraphs)
#     try:
#         content = np.load(EMBEDDINGS_FILE, allow_pickle=True)
#     except Exception as e:
#         return f"Error loading stored text: {e}"

#     # Encode query
#     query_embedding = model.encode([normalized_question], convert_to_numpy=True, normalize_embeddings=True)

#     # Perform FAISS search (Cosine Similarity via Inner Product)
#     distances, indices = index.search(query_embedding, top_k)
    
#     # Debugging FAISS search results
#     print(f"DEBUG - FAISS Distances: {distances}")
#     print(f"DEBUG - FAISS Indices: {indices}")

#     # Convert FAISS Inner Product similarity to Cosine Similarity
#     cosine_similarities = 1 - distances  # Since FAISS returns distances

#     # If no relevant results found
#     if indices is None or len(indices[0]) == 0:
#         return "No relevant results found."

#     # Define similarity threshold (higher = more strict)
#     SIMILARITY_THRESHOLD = 0.3  # Adjust for best accuracy
#     filtered_chunks = []

#     # Filter chunks based on similarity score
#     for sim, idx in zip(cosine_similarities[0], indices[0]):
#         if sim >= SIMILARITY_THRESHOLD and 0 <= idx < len(content):
#             filtered_chunks.append((content[idx], sim))

#     # If no strong matches, return fallback message
#     if not filtered_chunks:
#         return "No strongly relevant results found."

#     # Sort by highest similarity score
#     filtered_chunks.sort(key=lambda x: x[1], reverse=True)

#     # Extract just the text
#     best_chunks = [chunk[0] for chunk in filtered_chunks]

#     # Keyword-based filtering for better accuracy
#     must_have_keywords = ["student", "add", "register", "enroll", "update","child","promote","dashboard","CMS","login","attendance"]
#     refined_chunks = [chunk for chunk in best_chunks if any(word in chunk.lower() for word in must_have_keywords)]

#     # If keyword filtering removes all results, return original filtered results
#     if not refined_chunks:
#         refined_chunks = best_chunks  

#     # Remove duplicates & very short responses
#     refined_chunks = list(dict.fromkeys(refined_chunks))  # Remove duplicates
#     refined_chunks = [chunk for chunk in refined_chunks if len(chunk.split()) > 10]  # Ensure meaningful responses

#     # If still empty after filtering
#     if not refined_chunks:
#         return "No strongly relevant results found."

#     # Summarize text for better readability
#     joined_text = " ".join(refined_chunks)
#     summarized_response = summarize_text(joined_text)

#     # Adjust wording based on query
#     final_response = adjust_response_wording(summarized_response, query)

#     # Return response with greeting if applicable
#     return f"{get_greeting_response(greeting)}\n\n{final_response}" if greeting else final_response


#QUERY
# Example search
# query = "hi how to add a student"
# results = search_faiss(query, top_k=3)
# print("Search Results:\n", results)


#Loader
# import os
# import faiss
# import numpy as np
# from sentence_transformers import SentenceTransformer


# # Load Sentence Transformer model
# print("Loading Sentence Transformer model...")
# model = SentenceTransformer("all-MiniLM-L6-v2")
# print("Model loaded successfully!")


# def load_scraped_content(chunk_size=3, overlap=1):
#     """Loads the scraped content and splits it into smaller chunks."""
#     if not os.path.exists(STORAGE_FILE):
#         print(f"Error: {STORAGE_FILE} does not exist.")
#         return []

#     with open(STORAGE_FILE, "r", encoding="utf-8") as file:
#         lines = [line.strip() for line in file.readlines() if line.strip()]

#     # Split text into small overlapping chunks
#     chunks = []
#     # Overlapping chunks
#     for i in range(0, len(lines), chunk_size - overlap):  
#         # Merge lines into a chunk
#         chunk = " ".join(lines[i:i + chunk_size])  
#         if chunk:
#             chunks.append(chunk)

#     print(f"Loaded {len(chunks)} text chunks from {STORAGE_FILE}.")
#     return chunks


# def build_faiss_index():
#     """Builds and saves a FAISS index using sentence embeddings."""
#     print("\nLoading scraped content...")
#     content = load_scraped_content()
#     if not content:
#         print("No content found. Exiting FAISS index creation.")
#         return

#     print("Generating embeddings...")
#     try:
#         embeddings = model.encode(content, convert_to_numpy=True, normalize_embeddings=True)
#         print(f"Embeddings generated with shape: {embeddings.shape}")
#     except Exception as e:
#         print(f"Error generating embeddings: {e}")
#         return

#     # Remove old index if it exists
#     if os.path.exists(FAISS_INDEX_FILE):
#         print("Deleting old FAISS index...")
#         os.remove(FAISS_INDEX_FILE)

#     if os.path.exists(EMBEDDINGS_FILE):
#         print("Deleting old embeddings file...")
#         os.remove(EMBEDDINGS_FILE)

#     # Create FAISS index
#     dimension = embeddings.shape[1] if embeddings.shape[0] > 0 else None
#     if dimension is None:
#         print("Error: No embeddings to index.")
#         return

#     print(f"Creating FAISS index with dimension: {dimension}")
#     index = faiss.IndexFlatIP(dimension)
    
#     # Adding embeddings to FAISS index
#     print("Adding embeddings to FAISS index...")
#     index.add(embeddings)

#     if index.ntotal == 0:
#         print("Error: No vectors added to FAISS index.")
#         return
#     else:
#         print(f"Total vectors in FAISS index: {index.ntotal}")

#     # Save FAISS index and text chunks
#     print("Saving FAISS index and embeddings...")
#     faiss.write_index(index, FAISS_INDEX_FILE)
#     np.save(EMBEDDINGS_FILE, np.array(content, dtype=object))

#     print("FAISS index built and saved successfully.")


# # Run FAISS index builder
# if __name__ == "__main__":
#     print("\nStarting FAISS index building process...")
#     build_faiss_index()
