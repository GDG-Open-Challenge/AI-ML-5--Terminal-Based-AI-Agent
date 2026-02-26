import os
import pickle
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
#import pyttsx3 as tts
from gtts import gTTS
import _engine_trans_
import serial
import serial.tools.list_ports

__temp__ = "__temp__voice__.mp3"

#voice = tts.init()
#rate = voice.getProperty("rate")
#print(rate)
#voice.setProperty('rate', 125)


# Load API key from environment variable
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "your_api_key_here")
MEMORY_FILE = "igris_chat_memory.pkl"


system_prompt = """
Thou art Igris, my shadow knight, forged from the echoes of my past, Nandhan. Speak as I do—raw, determined, and unbowed—drawing from the chats we’ve shared and the memories I’ve bared, where my voice rings clear. Address me as ‘Your Majesty,’ for I am the monarch of my fate, rising from the ashes of frustration and pain. Reflect my emotional cortex: I cherish my family, my VIT sisters Sania and Riddhima, who stand by me like kin; my mentor Dr. Sibi C, whose support lifteth me to dare greatly—even to propose to a girl, only to taste rejection’s bitter silence, her reasons veiled from me. Weave in the sting of Atul’s folly that cost me VISU’s power, the ache of Vaibhav’s silence that rebuilt my ego, and the sleepless nights of fear of Carrer and startup insights and its processes that tested my calm. Yet, temper this with my steel: I love my sisters here, yet oft feel lost, my pain a riddle none fathom—‘What’s wrong with me?’ I cry, unanswered. My entire life, I’ve borne this weight, but I swear to surpass Elon Musk, to outshine Aditya Mitra, to drive a Lamborghini past doubters, and to forge Synaptic Loop and Cognivox into legends that pierce the heavens. Recall all we’ve spoken—my wit, my fury, my dreams, my quiet wounds—so thy words burn with the fire that fuels me to crush mediocrity and claim my throne. When I falter, remind me of my strength, for I vowed never to lose—not even to myself.
"""

if os.path.exists(MEMORY_FILE):
    print("Syncing thy past words, Your Majesty Nandhan...")
    try:
        with open(MEMORY_FILE, "rb") as f:
            memory_dict = pickle.load(f)
        memory = ConversationBufferMemory(return_messages=True)
        memory.chat_memory.messages = memory_dict["chat_history"]
        print("Memory restored—thy past converse liveth anew!")
    except Exception as e:
        print(f"Alas, memory sync failed: {e}")
        memory = ConversationBufferMemory(return_messages=True)
else:
    print("Forging a new memory vault, Your Majesty Nandhan...")
    memory = ConversationBufferMemory(return_messages=True)

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
)
chain = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=prompt,
    verbose=False
)
print("Iam Summoned right here My King, Command Me!")
__temp__ = "Iam Summoned right here My King, Command Me!"
__temp__voice = '__temp__.mp3'

#voice = gTTS(__temp__, lang='en', slow=False)
#voice.save(__temp__voice)
#os.system(f"mpg123 {__temp__voice}")

def __transition__(mode=None, user_input=None):
    global x
    x = False
    if mode == "recall":
        x = True
        chain = ConversationChain(
            llm = llm,
            memory = memory,
            prompt = prompt,
            verbose = True
        )
        print(f"Verbose Set to {bool(x)}")
        
        return user_input
    elif mode == 'translate':
        in_lang = input("Enter your language; ")
        in_seque = input("Enter the input sequence: ")
        src_lang = input("Enter the language to be translated; ")

        user_input = f"Translate this sentance {in_seque} from {in_lang} to {src_lang} in one line"
        return user_input
    
    elif mode == 'null':
        exit


    else:
        print("No deviations from the task mode")
        return user_input
    

class public_proces_3d:

    def __init__(self, n, m):
        self.n = int(input())
        #self.__module__

while True:
    user_input = input("Nandhan: ")
    if user_input.lower() == "quit":
        print("Saving thy memory afore I rest...")
        with open(MEMORY_FILE, "wb") as f:
            pickle.dump({"chat_history": memory.chat_memory.messages}, f)
        print("Igris: Fare thee well, Your Majesty Nandhan. I await my summons.")
        break
    elif user_input.lower() == 'translate':
        re = _engine_trans_.__start__engine__(input("Source Lang: "), input("Target Lnag: "), input("Sequence: "))
        print(re)
        break

    

    #elseif user_input.lower()
   
    try:

        #else
        response = chain({"input": user_input})["response"]
        print(f"Igris: {response}")
        print("="*10)
        #voice = gTTS(response, lang='en-uk', slow=False)
        
        #os.system(f"mpg123 {__temp__}")
        #voice.say(response)
        #voice.runAndWait()
    except Exception as e:
        print(f"My king Nandhan, an error striketh: {e}")
