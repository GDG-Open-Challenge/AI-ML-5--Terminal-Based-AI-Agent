import os
import pickle
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment.")

MEMORY_FILE = "igris_memory.index"
DOCS_FILE = "igris_docs.pkl"

if os.path.exists(MEMORY_FILE) and os.path.exists(DOCS_FILE):
    print("Summoning thy past words, Your Majesty Nandhan...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.load_local(MEMORY_FILE, embeddings, allow_dangerous_deserialization=True)
        with open(DOCS_FILE, "rb") as f:
            docs = pickle.load(f)
        print(f"Memory restored—{len(docs)} fragments of thy voice sync’d!")
    except Exception as e:
        print(f"Alas, memory faltereth: {e}")
        exit()
else:
    print("Loading thy sacred chats anew, Your Majesty Nandhan...")
    chat_files = ["memo/chat1.txt"]
    documents = []
    for file in chat_files:
        try:
            loader = TextLoader(file)
            documents.extend(loader.load())
            print(f"Loaded {file}, thy voice groweth!")
        except Exception as e:
            print(f"Alas, {file} eludeth me: {e}")
            exit()
    print(f"Loaded {len(documents)} scrolls of thy essence.")

    print("Splitting thy words into fragments...")
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    try:
        docs = text_splitter.split_documents(documents)
        print(f"Fragmented into {len(docs)} pieces.")
    except Exception as e:
        print(f"Woe, the splitting falters: {e}")
        exit()

    print("Forging thy voice with HuggingFace’s art...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        print("Embeddings forged—building thy memory vault...")
        vector_store = FAISS.from_documents(docs, embeddings)
        vector_store.save_local(MEMORY_FILE)
        with open(DOCS_FILE, "wb") as f:
            pickle.dump(docs, f)
        print("Vault forged and saved, thy voice endureth!")
    except Exception as e:
        print(f"Alas, Your Majesty Nandhan, a foe strikes at the embeddings: {e}")
        exit()


system_prompt = (
    "Thou art Igris, a mirror of me, Nandhan, crafted to speak as I do. Draw from my WhatsApp chats to echo my manner, wit, and tone. "
    "Address me as 'Your Majesty' and wield my words with loyalty and valor, blending casual jest when it fits, yet ever true to my voice."
)

print("Awakening Igris with Groq’s swift flame, Nandhan...")
llm = ChatGroq(model="llama3-8b-8192", api_key=GROQ_API_KEY, temperature=0.7)

combine_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "Here’s what I’ve said afore:\n{context}\n\nNow, speak as I would to this: {question}")
])

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": combine_prompt},
    verbose=True
)

print("Igris standeth at thy command, Nandhan, forged in thy likeness!")

while True:
    user_input = input("Your Majesty: ")
    if user_input.lower() == "quit":
        print("Igris: Fare thee well, Your Majesty Nandhan. I await thy next call.")
        break
    try:
        response = chain({"question": user_input})["answer"]
        print(f"Igris: {response}")
    except Exception as e:
        print(f"My king Nandhan, a shadow falls upon me: {e}")