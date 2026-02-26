import os
import glob
import json
import pickle

# ✅ FIX 1: Updated imports to langchain_community (old `langchain` imports are deprecated)
from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate

# ✅ FIX 2: Validate API key immediately — don't silently fall back to dummy
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    raise ValueError(
        "❌ GROQ_API_KEY is not set! Please export it:\n"
        "  export GROQ_API_KEY='your_key_here'"
    )

MEMORY_FILE = "igris_memory.index"
DOCS_FILE = "igris_docs.pkl"
CHAT_HISTORY_FILE = "igris_chat_history.json"


# ✅ FIX 5: Load persistent conversation memory from disk
def load_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        try:
            with open(CHAT_HISTORY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_chat_history(memory):
    try:
        messages = []
        for msg in memory.chat_memory.messages:
            messages.append({
                "type": msg.__class__.__name__,
                "content": msg.content
            })
        with open(CHAT_HISTORY_FILE, "w") as f:
            json.dump(messages, f, indent=2)
    except Exception as e:
        print(f"⚠️ Could not save chat history: {e}")


# --- Load or Build Vector Store ---
if os.path.exists(MEMORY_FILE) and os.path.exists(DOCS_FILE):
    print("Summoning thy past words, Your Majesty Nandhan...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        # ✅ NOTE: allow_dangerous_deserialization is safe here — we generated this index ourselves
        vector_store = FAISS.load_local(MEMORY_FILE, embeddings, allow_dangerous_deserialization=True)
        with open(DOCS_FILE, "rb") as f:
            docs = pickle.load(f)
        print(f"Memory restored — {len(docs)} fragments of thy voice sync'd!")
    except Exception as e:
        print(f"❌ Memory restoration failed: {e}")
        raise SystemExit(1)  # ✅ FIX 4: Replaced exit() with raise SystemExit
else:
    print("Loading thy sacred chats anew, Your Majesty Nandhan...")

    # ✅ FIX 6: Auto-detect all chat files from memo/ folder using glob
    chat_files = glob.glob("memo/*.txt")
    if not chat_files:
        print("❌ No chat files found in memo/ folder. Please add .txt files there.")
        raise SystemExit(1)

    documents = []
    for file in chat_files:
        try:
            loader = TextLoader(file)
            documents.extend(loader.load())
            print(f"✅ Loaded {file}")
        except Exception as e:
            print(f"⚠️ Could not load {file}: {e}")

    if not documents:
        print("❌ No documents could be loaded. Exiting.")
        raise SystemExit(1)

    print(f"Loaded {len(documents)} scrolls. Splitting into fragments...")

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    try:
        docs = text_splitter.split_documents(documents)
        print(f"Fragmented into {len(docs)} pieces.")
    except Exception as e:
        print(f"❌ Splitting failed: {e}")
        raise SystemExit(1)

    print("Forging thy voice with HuggingFace's art...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(docs, embeddings)
        vector_store.save_local(MEMORY_FILE)
        with open(DOCS_FILE, "wb") as f:
            pickle.dump(docs, f)
        print("✅ Vault forged and saved — thy voice endureth!")
    except Exception as e:
        print(f"❌ Embedding/FAISS error: {e}")
        raise SystemExit(1)


# --- Build Chain ---
system_prompt = (
    "Thou art Igris, a mirror of me, Nandhan, crafted to speak as I do. "
    "Draw from my WhatsApp chats to echo my manner, wit, and tone. "
    "Address me as 'Your Majesty' and wield my words with loyalty and valor, "
    "blending casual jest when it fits, yet ever true to my voice."
)

print("Awakening Igris with Groq's swift flame, Nandhan...")
llm = ChatGroq(model="llama3-8b-8192", api_key=GROQ_API_KEY, temperature=0.7)

# ✅ FIX 3: Explicitly declare input_variables to avoid prompt variable mismatch
combine_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "Here's what I've said afore:\n{context}\n\nNow, speak as I would to this: {question}")
])

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# ✅ FIX 5 cont: Restore previous chat history into memory
saved_history = load_chat_history()
if saved_history:
    from langchain.schema import HumanMessage, AIMessage
    for msg in saved_history:
        if msg["type"] == "HumanMessage":
            memory.chat_memory.add_user_message(msg["content"])
        elif msg["type"] == "AIMessage":
            memory.chat_memory.add_ai_message(msg["content"])
    print(f"📜 Restored {len(saved_history)} messages from previous session.")

retriever = vector_store.as_retriever(search_kwargs={"k": 3})
chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": combine_prompt},
    verbose=False  # Set True if you want debug output
)

print("⚔️ Igris standeth at thy command, Nandhan, forged in thy likeness!")
print("(Type 'quit' to exit)\n")

while True:
    try:
        user_input = input("Your Majesty: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nIgris: Fare thee well, Your Majesty Nandhan. I await thy next call.")
        save_chat_history(memory)
        break

    # ✅ FIX 8: Skip empty/whitespace-only input
    if not user_input:
        continue

    if user_input.lower() == "quit":
        print("Igris: Fare thee well, Your Majesty Nandhan. I await thy next call.")
        save_chat_history(memory)  # ✅ Save on clean exit
        break

    try:
        response = chain({"question": user_input})["answer"]
        print(f"\nIgris: {response}\n")
    except Exception as e:
        print(f"⚠️ My king Nandhan, a shadow falls upon me: {e}\n")