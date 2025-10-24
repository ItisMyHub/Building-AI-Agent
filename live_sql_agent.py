# This is the main file for my Showcase 2. It uses LangChain's modern agent toolkit to let the AI directly query the database to answer questions.

# These are the key components from the LangChain library.
from langchain_community.chat_models import ChatOllama 
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
import os

# --- Configuration ---
DB_PATH = "analytics.db"
# This gets the model name from an environment variable, with 'llama3.2' as a default.
MODEL_NAME = os.environ.get("RAG_LLM_MODEL", "llama3.2")

# --- Agent Setup ---
def query_live_database(user_query: str):
    """
    Takes a user's query, runs it through the modern SQL agent,
    and returns the final answer.
    """
    print(f"\n--- Received Query for SQL Agent: '{user_query}' ---")
    try:
        # 1. Initialize the LLM as a CHAT model, and here chatollama is implemented 
        llm = ChatOllama(model=MODEL_NAME, temperature=0)
        # 2. Connecting to the database
        db_uri = f"sqlite:///{DB_PATH}"
        db = SQLDatabase.from_uri(db_uri)
        
        # 3. The 'create_sql_agent' function here bundles the LLM and the database connection into an intelligent agent that can write and execute its own SQL queries.
        agent_executor = create_sql_agent(
            llm=llm,
            db=db,
            verbose=True, # Through this I can see the agent's thought process in my terminal. This has helped debugging in some cases.
            agent_type="openai-tools"
        )

        # 4. Invoking the agent with the user query
        result = agent_executor.invoke({"input": user_query})
        
        # The answer is now in the 'output' key
        final_answer = result.get("output", "Sorry, I was unable to process that request.")
        
        print(f"--- Final Answer: {final_answer} ---")
        return {"answer": final_answer}

    except Exception as e:
        print(f"An error occurred in the SQL agent: {e}")
        return {"answer": f"An error occurred: {e}"}

# --- Example of how to run it directly ---
if __name__ == "__main__":
    print("--- Live Database Information Assistant (Modern) ---")
    query_live_database("How much is the total revenue from all orders?")