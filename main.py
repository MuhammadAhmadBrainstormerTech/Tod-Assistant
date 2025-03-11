import os
from bs4 import BeautifulSoup
import re
import random
from flask import Flask, request, jsonify
import requests
from fuzzywuzzy import fuzz, process
import nltk
from sumy.parsers.plaintext import PlaintextParser
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from flask import Flask, request, jsonify
import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('punkt')
from sentence_transformers import SentenceTransformer


# File paths
STORAGE_FILE = "scraped_content.txt"
FAISS_INDEX_FILE = "faiss_indexx.bin"
EMBEDDINGS_FILE = "embeddingss.npy"

# Load Sentence Transformer model
print("Loading Sentence Transformer model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded successfully!")


GREETINGS = [
    "Hey there! 😊",  
    "Hello! How are you Today? 😊",  
    "Hi! Hope you're doing great.",  
    "Hey! Let’s explore your questions together 🚀",  
    "Hi there! 💡",  
    "Hello! Hope your day is going well. ✨",  
    "Hey🔍",
    "Hi!👍",  
    "Hey there! Ready for a quick assistance 😊",  
    "Hello! Let’s dive in! 🏆",  
    "Hi, I'm Todd, your friendly assistant 😊", 
    "Hi there! I'm good 😊",  
    "Doing great! How can I help? 👍",  
    "I'm good, thanks for asking! What about you?",  
    "Nice to meet you! How can I assist? 🤖",  
    "Hope you're having a great day! ☀️",  
    "Hey! What's new with you? 🚀",  
    "It's great to see you! 😊",  
]

CLOSINGS = [
    "Hope this helps! Let me know if you need anything else. 🚀",
    "That's all for now! Let me know if you have more questions. 😊",
    "Feel free to ask if you need further clarification. Have a great day! 🎉",
    "Let me know if I can assist you with anything else! ✨",
    "I hope this makes things clearer. Reach out if you need more help! 👍",
    "If you need any more details, just let me know! 😊",
    "That’s it for now! Hope you have a great day ahead. 🌟",
    "Happy to help! Let me know if you need more info. 🎯",
    "Hope this answered your question! Feel free to ask anything else. 🔍",
    "Take care and let me know if you need anything else! 😊"
]

KEYWORD_EMOJI_MAP = {
    "click": "🖱️", "select": "✅", "navigate": "🧭", "update": "🔄", "manage": "📋",
    "open": "📂", "go to": "➡️", "press": "🔘", "drag": "🖱️", "drop": "📥",
    "choose": "🎯", "install": "⚙️", "enable": "🔛", "disable": "🚫",
    "configure": "🔧", "customize": "🎨"
}

APPRECIATION_RESPONSES = [
    "You're welcome! 😊 Let me know if you need anything else.",
    "Glad I could help! 🚀",
    "Happy to assist! 🎯",
    "No problem! Let me know if you have more questions. 👍",
    "Always here to help! 😊",
    "Thanks for your kind words! Let me know if you need more info. 💡",
    "Much appreciated! If you need anything else, feel free to ask! ✨"
]
HELP_RESPONSE = [
    # Friendly & Encouraging Responses
    "Sure!! What do you need help with? 😊",
    "Of course! I'm happy to help. What's your question? 🚀",
    "No worries! Tell me what you need help with, and I'll do my best! ✨",
    "Absolutely! Let me know what you're stuck on. I'll guide you. 💡",
    "You're not alone! I'm here to help. What do you need assistance with? 🔍",

    # Professional & Supportive Responses
    "I'm here to assist. Please describe your issue, and I'll help you solve it. 👍",
    "I'm happy to support you. Could you provide more details about your question? 🤔",
    "I’d be glad to help! Let me know what you're struggling with. 🎯",
    "No problem! Just let me know how I can assist you today. 🔧",
    "I understand! Please share the details, and I'll do my best to help. ✅",

    # Fast & Reassuring Responses
    "Got it! Let’s tackle this together. What do you need help with? 💪",
    "I see! Let me simplify things for you. Just tell me what you need. 📝",
    "Helping is what I do best! What’s on your mind? 🎉",
    "I hear you! Let’s find the best solution together. Tell me more. 🔎",
    "Sure thing! Let me guide you step by step. What’s the issue? 👨‍🏫",

    # Problem-Solving & Guidance-Based Responses
    "Don’t worry, we’ll figure this out! Just give me the details. 🚀",
    "Let's get this sorted! Tell me what's going on, and I'll help. 🎯",
    "I’m ready to assist! What specific issue are you facing? 🛠",
    "Break it down for me! I’ll help you get to the solution. 🧩",
    "Happy to help! Let's find the best way forward. What’s the challenge? ⚡",

    # Empathetic & Motivational Responses
    "I know how frustrating this can be! Let’s work through it together. 🌟",
    "Take your time! I’ll be here to help whenever you’re ready. ⏳",
    "You're doing great! Let’s make this easier. What do you need? 🚀",
    "I totally get it! Let’s get this sorted right away. 🔥",
    "You're not alone in this! Let's solve it step by step. 🏆"
]

GENERIC_GREETINGS = {
    # ✅ Common English greetings (Mapped to Multiple Responses)
    ("hi", "hello", "hey", "hey there", "hi there", "greetings", "hello there"): [0, 2, 8, 16], 
    ("good morning", "good afternoon", "good evening", "good day"): [5, 15, 17],  

    # ✅ "How are you?" Variations (Multiple Responses)
    ("how are you", "how’s it going", "how are you doing", "how do you do", 
     "how have you been", "how’s your day", "how’s life", "how’s everything"): [3, 11, 12, 13],  

    # ✅ Casual slang & internet greetings (Mapped to Multiple Responses)
    ("wassup", "whatsup", "sup", "sup bro", "sup dude", "yo", "hiya", "holla", 
     "hey mate", "yo yo", "what’s good", "what’s new", "hey fam", "hey bro", "hey sis", "how’s it hanging"): [6, 7, 16],  

    # ✅ Friendly reunion greetings (Mapped to Multiple Responses)
    ("long time no see", "hey friend", "hi buddy", "hello friend", "nice to meet you", "pleased to meet you"): [8, 14, 17],  

    # ✅ Formal greetings (Mapped to Multiple Responses)
    ("good to see you", "hope you’re doing well", "it's a pleasure to meet you"): [4, 5, 14],  

    # ✅ Multilingual greetings (Mapped to Multiple Responses)
    ("hola", "bonjour", "ciao", "shalom", "salam", "aloha", "namaste", "konnichiwa", 
     "annyeong", "ni hao", "guten tag", "privet", "zdravstvuyte", "merhaba", "sawubona", "vanakkam", "yassas"): [1, 9, 15],  

    # ✅ Todd Introductions (Mapped to Multiple Responses)
    ("who are you", "what are you", "introduce yourself", "tell me about yourself"): [10]  
}


APPRECIATION_KEYWORDS = [
    # Common appreciation phrases
    "thanks", "ok", "thank you", "appreciate it", "great work", "nice job",
    "well done", "awesome", "amazing", "good job", "fantastic", "love it",
    "keep it up", "kudos", "much appreciated", "hats off", "respect",
    
    # Expressing gratitude casually
    "thanks a lot", "many thanks", "thanks so much", "thanks a ton",
    "thanks a bunch", "cheers", "big thanks", "huge thanks", "massive thanks",
    "thanks buddy", "thanks bro", "thank you so much", "thank you tons",
    
    # Formal expressions of gratitude
    "I truly appreciate it", "I'm grateful", "much obliged", "I'm in your debt",
    "thank you kindly", "I can't thank you enough", "eternally grateful",
    "sincere thanks", "profound gratitude", "heartfelt thanks",

    # Internet slang/modern appreciation
    "ty", "tysm", "thx", "thnx", "gracias", "danke", "merci", "arigato",
    "shukran", "shukriya", "obrigado", "grazie", "dhanyavad", "takk",
    "you rock", "you're the best", "big fan", "mad respect", "goat",
    
    # Compliments & positive feedback
    "amazing job", "superb work", "phenomenal", "excellent work",
    "brilliant work", "outstanding effort", "exceptional", "top-notch",
    "terrific", "impressive", "legendary", "mind-blowing"
]
HELP_KEYWORDS = [
    # Direct Help Requests
    "help", "please help", "can you help me", "help me", "assist me", 
    "i need help", "i need assistance", "can you assist me", "help needed",
    
    # Casual Help Requests
    "i'm stuck", "stuck here", "i can't figure this out", "i don't get it",
    "help me out", "need guidance", "can you support", "support needed",
    "i'm confused", "i need some guidance", "can you explain this",

    # Formal/Professional Help Requests
    "i have an issue", "i have a problem", "can you clarify", "need clarification",
    "i need your help", "can i ask something", "i have a query", 
    "can i ask a question", "i need some advice", "can you provide guidance",
    "can you help me understand this", "can you shed some light on this",

    # Task-Specific Help Requests
    "how do i do this", "how do i use this", "how do i solve this", 
    "what do i do next", "what should i do", "i'm not sure what to do",
    "i don't know how to proceed", "how do i proceed", 
    "explain this to me", "guide me through this",

    # Polite Help Requests
    "could you help me", "would you mind helping me", "i'd appreciate your help",
    "may i ask for help", "kindly assist me", "please assist me",
    "i would like some help", "i'm seeking guidance", "can you lend a hand"
]
# ✅ Define Query Synonym Mapping
QUERY_SYNONYMS = {
    "student": ["kid", "child", "new student", "pupil"],
    "add": ["register", "enroll", "create"],
    "remove": ["delete", "erase", "unregister"],
    "access": ["view","go to"],
    "update": ["edit", "modify", "change"],
    "password": ["credentials", "passcode", "login key"],
    "staff": ["instructor", "educator", "employee","worker"],    
    # Dashboard-related
    "dashboard": ["panel", "control center", "admin panel", "overview", "interface"],
    "company dashboard": ["business panel", "corporate overview", "organization dashboard"],
    # Attendance-related
    "attendance": ["presence", "check-in", "roll call", "participation", "time tracking"],
    # Branch-related
    "branch": ["division", "unit", "location", "office", "subdivision"],
    # Classroom-related
    "classroom": ["lecture hall", "learning space", "study room", "training room"],
    # Promotion-related
    "promotion": ["advancement", "upgrade", "progression", "elevation", "boost"],
    # Admission Query-related
    "admission query": ["enrollment request", "application inquiry", "registration query", "student admission"]
}

# Load FAISS index and embeddings once to optimize performance
if os.path.exists("faiss_indexx.bin") and os.path.exists("embeddingss.npy"):
    index = faiss.read_index("faiss_indexx.bin")
    content = np.load("embeddingss.npy", allow_pickle=True)
    print("Load Success")
else:
    index = None
    content = None
    
PROMPT_TEMPLATE = """
{greeting}

    {summary}
🔑 Key Steps:
{key_steps}

{closing}
"""
first_query = True  


def clean_summary_text(text):
    """Cleans and refines summary text by removing unnecessary words and rewording."""
    text = re.sub(r'\b(that|which|however|thus|therefore|hence|additionally|moreover|furthermore|consequently|nevertheless)\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(for example|such as|including|like)\b', 'e.g.', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

def extract_key_sentences(text, num_sentences=3):
    """Extracts key sentences using TF-IDF importance scoring and position-based ranking."""
    sentences = sent_tokenize(text)

    if len(sentences) <= num_sentences:
        return " ".join(sentences) 

    # ✅ Compute TF-IDF scores for words
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(sentences)
    sentence_scores = tfidf_matrix.sum(axis=1).A1  

    # ✅Rank
    ranked_sentences = sorted(
        enumerate(sentence_scores), key=lambda x: (x[1], -x[0]), reverse=True
    )

    # ✅Top
    best_sentences = [sentences[idx] for idx, _ in ranked_sentences[:num_sentences]]
    return " ".join(best_sentences)

def generate_summary(text, num_sentences=3):
    """Generates an improved summary using LSA summarization."""
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()

    summary_sentences = summarizer(parser.document, num_sentences)
    summary_text = " ".join(str(sentence) for sentence in summary_sentences)

    return summary_text

def reduce_repeated_chars(text):
    """Reduces excessive character repetition but allows up to 3 consecutive occurrences."""
    return re.sub(r'(.)\1{3,}', r'\1\1\1', text) 

def safe_extract_one(text, choices):
    """Handles cases where fuzzy matching might return None."""
    result = process.extractOne(text, choices, scorer=fuzz.ratio)
    return result if result else ("", 0)  # Avoid unpacking error

def extract_greeting_and_question(query):
    """Detects if the full query is a greeting, appreciation, or an actual question using fuzzy matching."""

    # ✅ Remove punctuation and convert to lowercase for better matching
    query_cleaned = re.sub(r'[^\w\s]', '', query.lower().strip())

    # ✅ Normalize
    query_cleaned = reduce_repeated_chars(query_cleaned)

    words = word_tokenize(query_cleaned)
    full_query = " ".join(words) 
    
    
    # ✅Closest Match
    best_greeting_match, greeting_score = safe_extract_one(full_query, [item for subset in GENERIC_GREETINGS.keys() for item in subset])
    best_appreciation_match, appreciation_score = process.extractOne(full_query, APPRECIATION_KEYWORDS, scorer=fuzz.ratio)
    best_help_match, help_score = process.extractOne(full_query, HELP_KEYWORDS, scorer=fuzz.ratio)

    # ✅Greeting
    if greeting_score >= 70:
        return best_greeting_match.capitalize(), None  

    # ✅Appreciation
    if appreciation_score >= 70:
        return "Appreciation", None  
    
    if help_score >= 70:
        return "Help", None

    # ✅Mixed Greeting
    for phrase_group in GENERIC_GREETINGS.keys():
        for phrase in phrase_group:
            if full_query.startswith(phrase + " "): 
                greeting = phrase.capitalize()
                # ✅ Remove greeting from query
                cleaned_query = full_query[len(phrase):].strip()  
                 # ✅ Return cleaned query or None
                return greeting, cleaned_query if cleaned_query else None 

    return None, query 

def get_greeting_response(user_input):
    """Returns an appropriate greeting response based on user input."""
    user_input = user_input.lower()

    for key_set, response in GENERIC_GREETINGS.items():
        if user_input in key_set:
            # ✅ Picks random index, fetches from GREETINGS
            return GREETINGS[random.choice(response)]  

    return "Hey! How can I assist you today? 😊" 


def clean_text(text):
    """Removes extra spaces, fixes phrasing, and makes text more readable."""
    # Remove extra spaces/newlines
    text = re.sub(r'\s+', ' ', text).strip()  
    text = re.sub(r'\bprovides quick access to\b', 'lets you quickly access', text, flags=re.IGNORECASE)
    text = re.sub(r'\ballows managing\b', 'helps manage', text, flags=re.IGNORECASE)
    text = re.sub(r'\bclick on\b', 'select', text, flags=re.IGNORECASE)
    return text

def extract_sentences(text):
    """Splits text into meaningful sentences."""
    text = clean_text(text)
    sentences = re.split(r'(?<=[.!?])\s+', text)  # Proper sentence splitting
    return [s.strip() for s in sentences if len(s.split()) > 3]  # Filter out very short sentences

def extract_relevant_section(extracted_text, query):
    """Extract the most relevant heading and its related paragraphs."""
    lines = extracted_text.split("\n")
    relevant_heading = None
    relevant_paragraphs = []

    for line in lines:
        # Detect headings (h1, h2, etc.)
        if line.lower().startswith(("h1:", "h2:", "h3:", "h4:", "h5:", "h6:")):
            relevant_heading = line  # Store the heading
        elif relevant_heading and len(line.split()) > 5:  # Ensure it's a paragraph
            if any(keyword in relevant_heading.lower() for keyword in query.lower().split()):
                relevant_paragraphs.append(line)

    return "\n".join(relevant_paragraphs) if relevant_paragraphs else extracted_text


def get_key_steps(sentences, max_steps=10):
    """Extracts action-oriented key steps with fixed emoji assignments."""
    steps = []
    
    for s in sentences:
        for keyword, emoji in KEYWORD_EMOJI_MAP.items():
            if keyword in s.lower():
                steps.append(f"{emoji} {s}")
                break  # Ensure a step gets only one emoji

    return "\n".join(steps[:max_steps]) if steps else "🔹 No key steps available."

def summarize_text(text, query=" ", include_greeting=True):
    """Generates a formatted summary with title, key steps, and optional greetings."""

    text = extract_relevant_section(text, query)  # Extract relevant sections
    sentences = extract_sentences(text)  # Convert to a list of sentences

    if not sentences:
        return "No valid content found."

    # ✅ Only include greeting if allowed
    greeting = random.choice(GREETINGS) if include_greeting else ""
    closing = random.choice(CLOSINGS)

    summary_paragraph = generate_summary(text, num_sentences=2)

    response = PROMPT_TEMPLATE.format(
        greeting=greeting,
        summary=summary_paragraph,  
        key_steps=get_key_steps(sentences),
        closing=closing
    )

    # ✅ Remove extra spaces or unwanted greetings if not needed
    response = response.strip()
    if not include_greeting:
         # Remove greeting if not required
        response = response.replace(greeting, "").strip() 
         # Clean up empty lines
        response = response.replace("\n\n", "\n") 

    return response

def normalize_query(query):
    """Replaces query words with their most common synonyms for better search accuracy."""
    words = query.lower().split()
    normalized_words = []
    
    for word in words:
        found_synonym = False
        for key, synonyms in QUERY_SYNONYMS.items():
            if word in synonyms or word == key:
                normalized_words.append(key)  # Use the standard word
                found_synonym = True
                break
        
        if not found_synonym:
            normalized_words.append(word)  # Keep original if no synonym found
    
    return " ".join(normalized_words)

def adjust_response_wording(response, query):
    """Replaces default response wording to match the user's query phrasing."""
    words_in_query = query.lower().split()
    adjusted_response = response

    for key, synonyms in QUERY_SYNONYMS.items():
        for synonym in synonyms:
            if synonym in words_in_query:
                # Replace the standard word in the response with the synonym found in the query
                adjusted_response = re.sub(rf'\b{key}\b', synonym, adjusted_response, flags=re.IGNORECASE)

    return adjusted_response


from flask import Flask, request, jsonify
import os
import numpy as np
import faiss
import random

app = Flask(__name__)

# Default value for FAISS search
TOP_K = 3

@app.route('/chat', methods=['POST'])
def search_faiss():
    """Flask API endpoint for FAISS search"""

    # ✅ Ensure request has Content-Type application/json
    if request.content_type != "application/json":
        return jsonify({"error": "Invalid request. Content-Type must be 'application/json'"}), 415

    # ✅ Handle empty request body
    if not request.data or request.data.strip() == b'':
        return jsonify({"error": "Empty request body. Please provide valid JSON."}), 400

    try:
        # ✅ Attempt to parse JSON safely
        data = request.get_json()
        if not data:
            return jsonify({"error": "Empty JSON body. Please provide valid JSON."}), 400
    except Exception as e:
        return jsonify({"error": f"Invalid JSON format: {str(e)}"}), 400

    # ✅ Extract the "query" field
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"message": "Please enter something"}), 400

    # ✅ Extract greeting and question
    greeting, question = extract_greeting_and_question(query)

    # ✅ Handle special cases (Appreciation & Help requests)
    if greeting == "Appreciation":
        return jsonify({"response": random.choice(APPRECIATION_RESPONSES)})

    if greeting == "Help":
        return jsonify({"response": random.choice(HELP_RESPONSE)})

    # ✅ Handle greeting-only messages
    if greeting and question is None:
        return jsonify({"response": get_greeting_response(greeting)})

    # ✅ Handle cases where no valid question is found
    if question is None:
        return jsonify({
            "response": f"{get_greeting_response(greeting)}\n\nHow can I assist you today?" if greeting else "How can I assist you today?"
        })

    # ✅ Normalize the query for better matching
    normalized_question = normalize_query(question)

    # ✅ Ensure FAISS index files exist before searching
    if not os.path.exists(FAISS_INDEX_FILE) or not os.path.exists(EMBEDDINGS_FILE):
        return jsonify({"response": "No FAISS index found. Please build the index first."})

    # ✅ Load FAISS index and stored content
    try:
        index = faiss.read_index(FAISS_INDEX_FILE)
    except Exception as e:
        return jsonify({"response": f"Error loading FAISS index: {e}"})

    try:
        content = np.load(EMBEDDINGS_FILE, allow_pickle=True)
    except Exception as e:
        return jsonify({"response": f"Error loading stored text: {e}"})

    # ✅ Convert query into an embedding and perform FAISS search
    query_embedding = model.encode([normalized_question], convert_to_numpy=True, normalize_embeddings=True)
    distances, indices = index.search(query_embedding, TOP_K)

    # ✅ Handle case where FAISS returns no results
    if indices is None or len(indices[0]) == 0:
        return jsonify({"response": "No relevant results found."})

    # ✅ Retrieve results based on FAISS indices
    results = [content[i] for i in indices[0] if 0 <= i < len(content)]

    # ✅ Remove duplicates and short responses
    results = list(dict.fromkeys(results))  # Remove duplicates
    results = [r for r in results if len(r.split()) > 10]  # Ensure meaningful responses

    if not results:
        return jsonify({"response": "Oops! I don't know the answer to this, Please Contact the Team"})

    # ✅ Generate summary from retrieved results
    summarized_response = summarize_text(" ".join(results), query=normalized_question, include_greeting=(greeting is not None))

    # ✅ Adjust response wording based on the query
    final_response = adjust_response_wording(summarized_response, query)

    # ✅ Combine greeting & final response if greeting exists
    formatted_response = {
        "response": f"{get_greeting_response(greeting)}\n\n{final_response}" if greeting else final_response
    }

    return jsonify(formatted_response)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
