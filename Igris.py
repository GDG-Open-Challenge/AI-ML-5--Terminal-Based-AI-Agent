import os
import pickle
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

from ingest_workflow import run_ingestion_workflow


GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

MEMORY_FILE = "igris_memory.index"
DOCS_FILE = "igris_docs.pkl"
DEFAULT_CHAT_FILES = ["memo/chat1.txt"]

def ensure_index_bootstrap():
    if os.path.exists(MEMORY_FILE) and os.path.exists(DOCS_FILE):
        return

    print("No memory index found. Bootstrapping from local chat files...")
    for file in DEFAULT_CHAT_FILES:
        if not os.path.exists(file):
            print(f"Skipping missing bootstrap file: {file}")
            continue
        result = run_ingestion_workflow(
            input_path=file,
            memory_file=MEMORY_FILE,
            docs_file=DOCS_FILE,
            bucket_name=None,
        )
        if result.get("error"):
            raise RuntimeError(result["error"])


def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(
        MEMORY_FILE,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def build_chain(llm, vector_store, memory, combine_prompt):
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": combine_prompt},
        verbose=True,
    )


ensure_index_bootstrap()
vector_store = load_vector_store()
with open(DOCS_FILE, "rb") as f:
    docs = pickle.load(f)
print(f"Memory restored—{len(docs)} fragments available.")


system_prompt = (
    "Thou art Igris, a mirror of me, Nandhan, crafted to speak as I do. Draw from my WhatsApp chats to echo my manner, wit, and tone. "
    "Address me as 'Your Majesty' and wield my words with loyalty and valor, blending casual jest when it fits, yet ever true to my voice."
)

print("Awakening Igris with Groq’s swift flame, Nandhan...")
if not GROQ_API_KEY:
    raise RuntimeError("Set GROQ_API_KEY before running.")
llm = ChatGroq(model="llama3-8b-8192", api_key=GROQ_API_KEY, temperature=0.7)

combine_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "Here’s what I’ve said afore:\n{context}\n\nNow, speak as I would to this: {question}")
])

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
chain = build_chain(llm=llm, vector_store=vector_store, memory=memory, combine_prompt=combine_prompt)
print("Igris standeth at thy command, Nandhan, forged in thy likeness!")
print("Commands: /ingest <path> [s3-bucket], /help, quit")


while True:
    user_input = input("Your Majesty: ")
    if user_input.lower() == "quit":
        print("Igris: Fare thee well, Your Majesty Nandhan. I await thy next call.")
        break
    if user_input.strip().lower() == "/help":
        print("Usage:")
        print("  /ingest <local_path> [s3_bucket]  -> read docs, index them, optional cloud upload")
        print("  quit                              -> exit")
        continue
    if user_input.strip().lower().startswith("/ingest "):
        parts = user_input.split()
        if len(parts) < 2:
            print("Igris: Provide a file/folder path. Example: /ingest docs/ my-bucket-name")
            continue
        ingest_path = parts[1]
        bucket = parts[2] if len(parts) > 2 else os.getenv("IGRIS_S3_BUCKET")
        print(f"Igris: Ingesting {ingest_path}...")
        result = run_ingestion_workflow(
            input_path=ingest_path,
            memory_file=MEMORY_FILE,
            docs_file=DOCS_FILE,
            bucket_name=bucket,
            cloud_prefix="igris",
        )
        if result.get("error"):
            print(f"Igris: Ingestion failed: {result['error']}")
            continue
        if result.get("cloud_uri"):
            print(f"Igris: Cloud upload complete -> {result['cloud_uri']}")
        else:
            print("Igris: Indexed locally. Cloud upload skipped.")
        vector_store = load_vector_store()
        chain = build_chain(llm=llm, vector_store=vector_store, memory=memory, combine_prompt=combine_prompt)
        continue

    try:
        response = chain({"question": user_input})["answer"]
        print(f"Igris: {response}")
    except Exception as e:
        print(f"My king Nandhan, a shadow falls upon me: {e}")
