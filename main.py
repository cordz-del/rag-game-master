# main.py
import streamlit as st
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# Initialize OpenAI LLM
llm = OpenAI()

# Initialize conversation memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize Chroma vector store
vector_store = Chroma(collection_name="rag_game_master", embedding_function=OpenAIEmbeddings())

# Define the prompt template for the game master
prompt_template = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="""
    You are a game master for a role-playing game. Respond to the player's questions based on the game context and previous chat history.
    Chat History:
    {chat_history}
    Player's Question: {question}
    """
)

# Initialize Conversational Retrieval Chain
chain = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=vector_store.as_retriever(),
    memory=memory,
    get_chat_history=lambda h: h,
    combine_docs_chain_kwargs={"prompt": prompt_template}
)

# Streamlit app interface
st.title("RAG Game Master")

# Input for player's question
question = st.text_input("Ask the Game Master:")

# Handle player's question and generate response
if question:
    response = chain.run(question)
    st.write(f"Game Master: {response}")
