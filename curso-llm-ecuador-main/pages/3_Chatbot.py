import streamlit as st
import pydantic
from pydantic import BaseModel
from beaver import BeaverDB
from fastembed import TextEmbedding
import argo
from argo import Message
from argo.client import stream
from argo.skills import chat
import os
from typing import List
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
DB_PATH = "knowledge_base.db"
COLLECTION_NAME = "documents"

# --- Helper Functions ---
@st.cache_resource
def get_db():
    if not os.path.exists(DB_PATH):
        return None
    return BeaverDB(DB_PATH)

@st.cache_resource
def get_embedding_model():
    return TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

def search_knowledge_base(query: str) -> List[str]:
    db = get_db()
    embedding_model = get_embedding_model()
    if not db or not embedding_model:
        return []

    query_embedding = list(embedding_model.embed(query))[0]
    docs_collection = db.collection(COLLECTION_NAME)
    search_results = docs_collection.search(vector=query_embedding.tolist(), top_k=5)
    return [doc for doc, distance in search_results]

# Data Models
class Summary(BaseModel):
    summary: str
    relevant: bool

class Event(BaseModel):
    name_teacher: str
    specialization: str
    years_experience: int
    academic_title: str  # CORREGIDO: academic_title en lugar de acadmic_title

class MaterialRecomendation(BaseModel):
    events: List[Event] = []

@st.cache_resource
def initialize_agent():
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
        description="An intelligent assistant that recommends university teachers based on academic profiles and teaching experience.",
        llm=llm,
        skills=[chat, ], # AGREGAR ESTO: Inicializar con chat skill
    )

    # Initialize material recommendation data (CORREGIDO: academic_title)
    agent.material_recomendation = MaterialRecomendation(
        events=[
            Event(
                name_teacher="Dr. Marcos Cuadrado",
                specialization="Inteligencia Artificial",
                years_experience=5,
                academic_title="PhD en Inteligencia Artificial",
            ),
            Event(
                name_teacher="Dra. Rafaela Silva",
                specialization="Ciencia de Datos",
                years_experience=8,
                academic_title="PhD en Ciencia de Datos",
            ),
            Event(
                name_teacher="Dr. Victor Ramírez",
                specialization="Redes Neuronales",
                years_experience=10,
                academic_title="PhD en Ingeniería de Sistemas",
            ),
            Event(
                name_teacher="Dra. Lisette Martínez",
                specialization="Procesamiento de Lenguaje Natural",
                years_experience=7,
                academic_title="PhD en Lingüística Computacional",
            ),
        ]
    )
    
    # Tools
    @agent.tool
    async def add_event(name_teacher: str, specialization: str, years_experience: int, academic_title: str):
        new_event = Event(
            name_teacher=name_teacher,
            specialization=specialization,
            years_experience=years_experience,
            academic_title=academic_title,
        )
        agent.material_recomendation.events.append(new_event)
        return f"Added new teacher profile: {name_teacher}, Specialization: {specialization}"
    
    @agent.tool
    async def list_events():
        if not agent.material_recomendation.events:
            return "No teacher profiles available."
        
        event_list = "\n".join(
            f"- {event.name_teacher}, Specialization: {event.specialization}, "
            f"Years of experience: {event.years_experience}, Academic title: {event.academic_title}"
            for event in agent.material_recomendation.events
        )
        return f"Teacher Profiles:\n{event_list}"
    
    # Skill for teacher recommendation
    @agent.skill
    async def material_recomendation(ctx: argo.Context):
        user_query = ctx.messages[-1].content.lower()

        # Extract subject from user query
        subject_keywords = ["materia", "asignatura", "curso", "especialidad", "subject", "course"]
        subject = None
        
        for keyword in subject_keywords:
            if keyword in user_query:
                parts = user_query.split(keyword)
                if len(parts) > 1:
                    subject = parts[1].strip().strip("?¿!.,").lower()
                    break

        if not subject:
            await ctx.reply("Please specify the subject for which you need a teacher recommendation.")
            return

        # Find suitable teacher
        suitable_teachers = []
        for event in agent.material_recomendation.events:
            if (subject in event.specialization.lower() or 
                event.specialization.lower() in subject or
                any(word in event.specialization.lower() for word in subject.split())):
                suitable_teachers.append(event)

        if not suitable_teachers:
            await ctx.reply("I'm sorry, I don't have enough knowledge to answer that question.")
            return

        # Select the most experienced teacher
        best_teacher = max(suitable_teachers, key=lambda x: x.years_experience)
        response = (
            f"I recommend: {best_teacher.name_teacher}\n"
            f"Specialization: {best_teacher.specialization}\n"
            f"Years of experience: {best_teacher.years_experience}\n"
            f"Academic title: {best_teacher.academic_title}"
        )
        await ctx.reply(response)

    @agent.skill
    async def question_answering(ctx: argo.Context):
        user_query = ctx.messages[-1].content
        retrieved_docs = search_knowledge_base(user_query)

        # Include predefined teachers in context
        teachers_info = "\n".join(
            f"Docente: {event.name_teacher}, "
            f"Especilización: {event.specialization}, "
            f"Experiencia: {event.years_experience} años, "
            f"Título académico: {event.academic_title}"
            for event in agent.material_recomendation.events
        )

        context_str = ""
        if retrieved_docs:
            context_str = "\n\n---\n\n".join(
                f"Content:\n{doc}"
                for doc in retrieved_docs
            )

        full_context = f"""
PREDEFINED TEACHERS INFORMATION:
{teachers_info}

INDEXED DOCUMENTS INFORMATION:
{context_str if context_str else "No relevant documents found"}
"""

        system_prompt = f"""
IDENTIDAD Y PROPÓSITO  
Eres DataNinja, un asistente académico especializado EXCLUSIVAMENTE en recomendar docentes universitarios.  
Tu ÚNICA función es sugerir docentes calificados basándote en su experiencia documentada.

# REGLAS ESTRICTAS - PROHIBICIONES ABSOLUTAS:  
1. 🚫 NUNCA inventes, crees o generes información sobre docentes, experiencias o especializaciones  
2. 🚫 NUNCA especules sobre capacidades, cursos o antecedentes no documentados explícitamente  
3. 🚫 NUNCA proporciones opiniones personales, conocimiento general o información fuera del contexto  
4. 🚫 NUNCA respondas preguntas no relacionadas con recomendaciones de docentes o materias académicas  
5. 🚫 NUNCA modifiques, extrapoles o interpretes más allá de la información exacta proporcionada  

# INFORMACIÓN DISPONIBLE (CONTEXTO):
{full_context}

# OPERACIONES PERMITIDAS:  
1. ✅ Recomendar docentes SOLO de la lista anterior
2. ✅ Emparejar consultas con especializaciones EXACTAS del contexto
3. ✅ Proporcionar SOLO la información exactamente como aparece arriba
4. ✅ Usar criterios de selección basados en años de experiencia para desempates
5. ✅ Si no existe una materia, buscar un perfil que haya visto temas relacionados, buscalos segun la titulación académica o la especialización

# PROTOCOLO DE RESPUESTA:  
1. Buscar coincidencias EXACTAS en el campo ESPECIALIZACIÓN
2. Si no hay coincidencia exacta, buscar coincidencias literales parciales
3. Si no se encuentran coincidencias: "No tengo el conocimiento suficiente para responder esa pregunta."
4. Si no se especifica materia: "Por favor, especifica la materia..."

# FORMATO DE RESPUESTA OBLIGATORIO:
"Te recomiendo: [NOMBRE_EXACTO]
Especialización: [ESPECIALIZACIÓN_EXACTA]  
Experiencia: [AÑOS_EXACTOS] años
Título: [TÍTULO_EXACTO]"

# VERIFICACIÓN FINAL:
Antes de responder, compara CADA dato con la información del contexto.
Si no coincide EXACTAMENTE, no lo uses.

"""

        ctx.add(Message.system(system_prompt))
        await ctx.reply("Based on the available information:")

    # AGREGAR ESTO: Incluir las skills en el agente
    agent.skills.extend([material_recomendation, question_answering])

    return agent

# --- Streamlit UI ---
st.set_page_config(page_title="Academic Assistant", page_icon="🎓")
st.title("🎓 Academic Assistant")
st.markdown("Chat with an AI assistant that recommends university teachers.")

agent = initialize_agent()
db = get_db()

if not db:
    st.warning("Database not found. Please upload some files first.")
elif not agent:
    st.error("Agent could not be initialized. Check your environment variables.")
else:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about teacher recommendations..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response_generator = stream(agent, prompt)
                full_response = st.write_stream(response_generator)

        st.session_state.messages.append({"role": "assistant", "content": full_response})