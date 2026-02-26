import os
import platform
import subprocess
from langchain.tools import Tool


###
def system_info(_=None):
    return f"""
OS: {platform.system()}
Version: {platform.version()}
Processor: {platform.processor()}
"""


###
def read_file(filename):
    try:
        with open(filename.strip(), "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"


###
def write_file(data):
    try:
        filename, content = data.split(":", 1)
        with open(filename.strip(), "w", encoding="utf-8") as f:
            f.write(content.strip())
        return f"File '{filename.strip()}' written successfully."
    except Exception as e:
        return f"Error writing file: {str(e)}"


###
def calculator(expression):
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Calculation error: {str(e)}"


###
def safe_terminal(command):
    allowed_commands = ["dir", "ls", "echo", "whoami", "date"]

    cmd = command.strip().split()[0]

    if cmd not in allowed_commands:
        return "Command not allowed for security reasons."

    try:
        result = subprocess.check_output(command, shell=True, text=True)
        return result
    except Exception as e:
        return f"Execution error: {str(e)}"


###
def create_retriever_tool(vector_store):

    def retrieve_memory(query):
        try:
            docs = vector_store.similarity_search(query, k=3)

            if not docs:
                return "No relevant memory found."

            return "\n\n".join([doc.page_content for doc in docs])

        except Exception as e:
            return f"Memory retrieval error: {str(e)}"

    return Tool(
        name="MemorySearch",
        func=retrieve_memory,
        description="Search past conversations and memory using semantic similarity"
    )


###
def get_tools(vector_store):

    retriever_tool = create_retriever_tool(vector_store)

    tools = [
        Tool(
            name="SystemInfo",
            func=system_info,
            description="Get system OS and processor details"
        ),

        Tool(
            name="ReadFile",
            func=read_file,
            description="Read contents of a file. Input should be filename"
        ),

        Tool(
            name="WriteFile",
            func=write_file,
            description="Write content to file. Format: filename:content"
        ),

        Tool(
            name="Calculator",
            func=calculator,
            description="Evaluate mathematical expressions"
        ),

        Tool(
            name="SafeTerminal",
            func=safe_terminal,
            description="Execute safe system commands like ls, dir, whoami, date, echo"
        ),

        retriever_tool
    ]

    return tools