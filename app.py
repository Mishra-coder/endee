import os
import streamlit as st
from dotenv import load_dotenv
from endee import Endee, Precision
from sentence_transformers import SentenceTransformer
from groq import Groq

load_dotenv()

try:
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception:
    pass

try:
    endee_client = Endee()
    endee_client.set_base_url("http://0.0.0.0:8080/api/v1")
except Exception:
    endee_client = None

INDEX_NAME = "internships"

INTERNSHIPS = [
    {"id": "1", "title": "Frontend Developer Intern", "company": "Google", "skills": "React JavaScript HTML CSS TypeScript frontend web development UI", "location": "Bangalore", "duration": "6 months", "stipend": "25000"},
    {"id": "2", "title": "Machine Learning Intern", "company": "Microsoft", "skills": "Python machine learning deep learning TensorFlow PyTorch AI neural networks data science", "location": "Hyderabad", "duration": "6 months", "stipend": "30000"},
    {"id": "3", "title": "Backend Developer Intern", "company": "Amazon", "skills": "Node.js Python Java REST API backend server database SQL microservices", "location": "Bangalore", "duration": "3 months", "stipend": "20000"},
    {"id": "4", "title": "Mobile App Developer Intern", "company": "Flipkart", "skills": "React Native Android iOS Flutter mobile app development JavaScript", "location": "Bangalore", "duration": "6 months", "stipend": "22000"},
    {"id": "5", "title": "Data Science Intern", "company": "Zomato", "skills": "Python data analysis pandas numpy matplotlib SQL data visualization statistics", "location": "Delhi", "duration": "3 months", "stipend": "18000"},
    {"id": "6", "title": "Full Stack Developer Intern", "company": "Swiggy", "skills": "React Node.js MongoDB Express MERN stack full stack JavaScript web development", "location": "Bangalore", "duration": "6 months", "stipend": "25000"},
    {"id": "7", "title": "DevOps Intern", "company": "Razorpay", "skills": "Docker Kubernetes AWS cloud CI/CD Linux deployment devops infrastructure", "location": "Bangalore", "duration": "6 months", "stipend": "28000"},
    {"id": "8", "title": "AI Research Intern", "company": "Adobe", "skills": "Python AI research NLP computer vision deep learning transformers BERT GPT", "location": "Noida", "duration": "6 months", "stipend": "35000"},
    {"id": "9", "title": "iOS Developer Intern", "company": "Paytm", "skills": "Swift iOS Xcode mobile development Apple UIKit SwiftUI", "location": "Noida", "duration": "3 months", "stipend": "20000"},
    {"id": "10", "title": "Android Developer Intern", "company": "CRED", "skills": "Kotlin Android Java mobile development Android Studio MVVM", "location": "Bangalore", "duration": "6 months", "stipend": "22000"},
    {"id": "11", "title": "Cybersecurity Intern", "company": "Infosys", "skills": "network security ethical hacking penetration testing Linux firewall cybersecurity", "location": "Pune", "duration": "3 months", "stipend": "15000"},
    {"id": "12", "title": "UI/UX Design Intern", "company": "Myntra", "skills": "Figma design UI UX user interface wireframe prototype design thinking", "location": "Bangalore", "duration": "3 months", "stipend": "15000"},
    {"id": "13", "title": "Blockchain Developer Intern", "company": "Polygon", "skills": "Solidity blockchain Ethereum Web3 smart contracts cryptocurrency decentralized", "location": "Remote", "duration": "6 months", "stipend": "30000"},
    {"id": "14", "title": "Data Engineer Intern", "company": "PhonePe", "skills": "Python SQL Spark Hadoop ETL pipeline data engineering big data warehouse", "location": "Bangalore", "duration": "6 months", "stipend": "25000"},
    {"id": "15", "title": "Product Management Intern", "company": "Meesho", "skills": "product management roadmap analytics user research agile scrum market research", "location": "Bangalore", "duration": "3 months", "stipend": "20000"},
]

def setup_index():
    if not endee_client: return
    try:
        endee_client.create_index(
            name=INDEX_NAME,
            dimension=384,
            space_type="cosine",
            precision=Precision.INT8,
        )
    except Exception:
        pass

def ingest_internships():
    if not endee_client: return
    try:
        index = endee_client.get_index(name=INDEX_NAME)
        vectors = []
        for internship in INTERNSHIPS:
            text = f"{internship['title']} {internship['company']} {internship['skills']} {internship['location']}"
            embedding = embedding_model.encode(text).tolist()
            vectors.append({
                "id": internship["id"],
                "vector": embedding,
                "meta": internship,
            })
        index.upsert(vectors)
    except Exception:
        pass

def search_internships(query: str, top_k: int = 5):
    if not endee_client: return []
    try:
        index = endee_client.get_index(name=INDEX_NAME)
        query_vector = embedding_model.encode(query).tolist()
        return index.query(vector=query_vector, top_k=top_k)
    except Exception:
        return []

def get_career_advice(query: str, matched_internships: str) -> str:
    try:
        prompt = f"""A student described themselves as: "{query}"
        Top matching internships found:
        {matched_internships}
        Give 3-4 lines of career advice: which internship suits them best and what skills they should improve."""
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    except Exception:
        return "Advice currently unavailable."

st.set_page_config(page_title="InternMatch AI", layout="wide")

if "indexed" not in st.session_state:
    setup_index()
    ingest_internships()
    st.session_state["indexed"] = True

st.title("InternMatch AI")
st.subheader("Find the perfect internship based on your skills - powered by Endee Vector DB")
st.markdown("---")

query = st.text_area(
    "Describe your skills and interests:",
    placeholder="e.g. I know Python, machine learning basics, and data analysis.",
    height=100,
)

search_btn = st.button("Find Matching Internships")

if search_btn and query:
    with st.spinner("Finding best matches..."):
        results = search_internships(query, top_k=5)

    if results:
        st.markdown("### Top Matching Internships:")
        context = ""
        for i, result in enumerate(results):
            meta = result.get("meta", {}) if isinstance(result, dict) else result.meta
            score = round(result.get("similarity", 0), 3) if isinstance(result, dict) else round(result.similarity, 3)
            context += f"{meta.get('title')} at {meta.get('company')}\n"
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"#### {i + 1}. {meta.get('title')} at {meta.get('company')}")
                st.write(f"Location: {meta.get('location')} | Duration: {meta.get('duration')} | Stipend: Rs {meta.get('stipend')}/month")
                st.write(f"Skills: {meta.get('skills')}")
            with col2:
                st.metric("Match Score", score)
            st.markdown("---")
        
        advice = get_career_advice(query, context)
        st.info(advice)
    else:
        st.warning("No matches found. Ensure the Endee server is running locally.")

st.caption("Built with Endee Vector DB + Sentence Transformers + Groq AI")
