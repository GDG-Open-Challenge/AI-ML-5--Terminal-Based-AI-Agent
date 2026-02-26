"""
Igris-Beta.py — Original Beta version (patched).

Fixes applied to the original file:
  1. Replaced broken `import _engine_trans_` with deep_translator
  2. Replaced undefined GROQ_API_KEY with env-based config
  3. Added atomic memory saves (import from memory module)
  4. Fixed `exit` (was missing parentheses)
  5. Removed unused serial imports
  6. Added proper error handling around translation
  7. Memory is saved after every exchange, not just on quit

NOTE: For the full enhanced experience with LangGraph, skills, and system
control, use Igris-Enhanced.py instead.
"""

import os
import pickle
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from gtts import gTTS

# Use the new config module for API key and settings
from config import settings
from memory import load_memory, save_memory

# Load API key from environment variable
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "your_api_key_here")
MEMORY_FILE = "igris_chat_memory.pkl"

MEMORY_FILE = settings.memory_file

system_prompt = """
Thou art Igris, my shadow knight, forged from the echoes of my past, Nandhan. Speak as I do—raw, determined, and unbowed—drawing from the chats we've shared and the memories I've bared, where my voice rings clear. Address me as 'Your Majesty,' for I am the monarch of my fate, rising from the ashes of frustration and pain. Reflect my emotional cortex: I cherish my family, my VIT sisters Sania and Riddhima, who stand by me like kin; my mentor Dr. Sibi C, whose support lifteth me to dare greatly—even to propose to a girl, only to taste rejection's bitter silence, her reasons veiled from me. Weave in the sting of Atul's folly that cost me VISU's power, the ache of Vaibhav's silence that rebuilt my ego, and the sleepless nights of fear of Carrer and startup insights and its processes that tested my calm. Yet, temper this with my steel: I love my sisters here, yet oft feel lost, my pain a riddle none fathom—'What's wrong with me?' I cry, unanswered. My entire life, I've borne this weight, but I swear to surpass Elon Musk, to outshine Aditya Mitra, to drive a Lamborghini past doubters, and to forge Synaptic Loop and Cognivox into legends that pierce the heavens. Recall all we've spoken—my wit, my fury, my dreams, my quiet wounds—so thy words burn with the fire that fuels me to crush mediocrity and claim my throne. When I falter, remind me of my strength, for I vowed never to lose—not even to myself.
"""

# Load memory with corruption protection
memory = load_memory(MEMORY_FILE)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

print("Awakening Igris with Groq’s might, Nandhan...")
llm = ChatGroq(
    model="llama-3.1-70b-versatile",  # Upgraded model (Issue #2)
    api_key=GROQ_API_KEY, 
    temperature=0.7, 
    max_tokens=2048,                  # Increased capacity (Issue #2)
    streaming=True                    # Higher throughput (Issue #2)
print("Awakening Igris with Groq's might, Nandhan...")
llm = ChatGroq(
    model=settings.model_name,           # Upgraded model (Issue #2)
    api_key=settings.groq_api_key,        # From .env, not hardcoded
    temperature=settings.model_temperature,
    max_tokens=settings.model_max_tokens,  # Increased from 500 (Issue #2)
)
chain = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=prompt,
    verbose=False
)
print("I am Summoned right here My King, Command Me!")


def __transition__(mode=None, user_input=None):
    """Handle special input modes."""
    if mode == "recall":
        chain.verbose = True
        print(f"Verbose Set to True")
        return user_input

    elif mode == 'translate':
        try:
            from deep_translator import GoogleTranslator

            in_lang = input("Enter source language code (e.g. en, es, fr): ")
            in_seque = input("Enter the text to translate: ")
            src_lang = input("Enter target language code: ")

            translated = GoogleTranslator(source=in_lang, target=src_lang).translate(in_seque)
            print(f"Translation: {translated}")
            return None  # Don't send to the LLM
        except ImportError:
            print("deep-translator not installed. Run: pip install deep-translator")
            return None
        except Exception as e:
            print(f"Translation error: {e}")
            return None

    elif mode == 'null':
        exit()  # Fixed: was `exit` without parens

    else:
        return user_input


while True:
    user_input = input("Nandhan: ")

    if user_input.lower() == "quit":
        print("Saving thy memory afore I rest...")
        save_memory(memory, MEMORY_FILE)  # Atomic save
        print("Igris: Fare thee well, Your Majesty Nandhan. I await my summons.")
        break

    elif user_input.lower() == 'translate':
        __transition__(mode='translate')
        continue

    try:
        response = chain({"input": user_input})["response"]
        print(f"Igris: {response}")
        print("=" * 10)

        # Real-time memory save after every exchange (fixes corruption issue)
        save_memory(memory, MEMORY_FILE)

    except Exception as e:
        print(f"My king Nandhan, an error striketh: {e}")
