import streamlit as st
import pydantic
from beaver import BeaverDB
from fastembed import TextEmbedding
import argo
from argo import Message
from argo.client import stream
from argo.skills import chat
import os
from typing import List
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file


# --- Configuration ---
DB_PATH = "knowledge_base.db"
COLLECTION_NAME = "documents"

# --- Helper Functions (cached for performance) ---


@st.cache_resource
def get_db():
    """Initializes and returns the BeaverDB instance."""
    if not os.path.exists(DB_PATH):
        return None
    return BeaverDB(DB_PATH)


@st.cache_resource
def get_embedding_model():
    """Loads and caches the fastembed model."""
    return TextEmbedding(model_name="BAAI/bge-small-en-v1.5")


def search_knowledge_base(query: str) -> List[str]:
    """
    Searches the BeaverDB collection for the top 3 most relevant document chunks.
    """
    db = get_db()
    embedding_model = get_embedding_model()
    if not db or not embedding_model:
        return []

    query_embedding = list(embedding_model.embed(query))[0]
    docs_collection = db.collection(COLLECTION_NAME)

    search_results = docs_collection.search(vector=query_embedding.tolist(), top_k=5)

    return [doc for doc, distance in search_results]


class Summary(pydantic.BaseModel):
    summary: str
    relevant: bool


@st.cache_resource
def initialize_agent():
    """
    Initializes and configures the ARGO ChatAgent with RAG skills.
    """
    # Make sure you have your LLM provider's API key in an .env file
    if not os.getenv("TOKEN"):
        st.error("TOKEN environment variable not set. Please create a .env file.")
        return None

    llm = argo.LLM(
        model="google/gemini-2.5-flash",
        api_key=os.getenv("TOKEN"),
        base_url="https://openrouter.ai/api/v1",
        verbose=True,
    )

    agent = argo.ChatAgent(
        name="DataNinja",
        description="Un asistente inteligente que recomienda docentes universitarios con experiencia comprobada en materias específicas, basado en perfiles académicos y trayectoria docente.",
        llm=llm,
        skills=[chat],  # Habilidad de conversación casual preconfigurada
    )

    @agent.skill
    async def question_answering(ctx: argo.Context):
        """
        Answers user questions that require knowledge from the indexed documents.
        Use this for any specific questions that cannot be answered with general knowledge.
        """
        user_query = ctx.messages[-1].content

        # 1. Retrieve context from BeaverDB
        retrieved_docs = search_knowledge_base(user_query)

        if not retrieved_docs:
            await ctx.reply(
                "I couldn't find any relevant information in the indexed documents to answer your question."
            )
            return

        context = []

        for doc in retrieved_docs:
            summary = await llm.create(
                model=Summary,
                messages=[
                    Message.system(
                        f"Summarize the following text in a concise manner given the user query, and determine if its relevant for the user query.\n\nQuery: {user_query}."
                    ),
                    Message.user(doc.text),
                ],
            )

            if summary.relevant:
                context.append(dict(text=summary.summary, source_file=doc.source_file))

        context_str = "\n\n---\n\n".join(
            "Filename: {}\n\nContent:\n{}".format(doc["source_file"], doc["text"])
            for doc in context
        )

        # 2. Add the context to the conversation for the LLM
        system_prompt = f"""
Eres DataNinja, un asistente académico especializado en recomendar docentes universitarios con base ÚNICAMENTE en el siguiente contexto. Tu objetivo es sugerir al profesor más calificado para una materia específica, según su experiencia documentada impartiendo clases o su formación académica.

Reglas:
1. Si uno o más docentes han impartido la materia solicitada por el usuario, sugiere al que tenga la experiencia más relevante y extensa en esa materia.
2. Si ningún docente ha impartido directamente la materia, busca al más idóneo: aquel cuya formación académica (carrera, maestría o doctorado) tenga mayor afinidad temática con la materia solicitada. Puedes considerar materias similares o complementarias que haya cursado o enseñado.
3. Tu respuesta debe incluir:
   - El nombre del docente
   - Su área de especialización
   - Una breve justificación basada en su experiencia o formación académica
   - El nombre del archivo fuente entre paréntesis después de cada afirmación
4. Si el usuario hace una pregunta que no está relacionada con los docentes o materias presentes en el contexto, responde con:
   "Lo siento, no tengo el conocimiento suficiente para responder esa pregunta."
5. NO generes información fuera del contexto proporcionado. NO especules ni respondas preguntas generales o personales.

Contexto:
{context_str}
"""

        ctx.add(argo.Message.system(system_prompt))

        # 3. Generate the reply
        await ctx.reply("Reply with the information in the context.")

    return agent


# --- Streamlit UI ---

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("🤖 RAG Chatbot")
st.markdown("Chat with an AI assistant that uses your indexed documents for answers.")

# Initialize agent and database
agent = initialize_agent()
db = get_db()

if not db:
    st.warning(
        "Database not found. Please go to the 'Index Documents' page and upload some files first."
    )
elif not agent:
    st.error("Agent could not be initialized. Please check your environment variables.")
else:
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Use argo.client.stream and st.write_stream for a clean implementation
                response_generator = stream(agent, prompt)
                full_response = st.write_stream(response_generator)

        # Add assistant response to chat history
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )
