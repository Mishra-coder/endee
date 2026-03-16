import os
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

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

@st.cache_resource
def build_vectorizer():
    texts = [f"{i['title']} {i['company']} {i['skills']} {i['location']}" for i in INTERNSHIPS]
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix

def search_internships(query, top_k=5):
    vectorizer, matrix = build_vectorizer()
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, matrix).flatten()
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(INTERNSHIPS[i], float(scores[i])) for i in top_indices]

def get_career_advice(query, matched):
    prompt = f'Student: "{query}"\nTop matches:\n{matched}\nGive 3-4 lines of career advice.'
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

st.set_page_config(page_title="InternMatch AI", layout="wide")
st.title("InternMatch AI")
st.subheader("Find the perfect internship based on your skills - powered by Endee Vector DB")
st.markdown("---")

query = st.text_area(
    "Describe your skills and interests:",
    placeholder="e.g. I know Python, machine learning basics, and data analysis.",
    height=100,
)

if st.button("Find Matching Internships") and query:
    with st.spinner("Finding best matches..."):
        results = search_internships(query)

    st.markdown("### Top Matching Internships:")
    context = ""

    for i, (internship, score) in enumerate(results):
        context += f"{internship['title']} at {internship['company']}\n"
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"#### {i+1}. {internship['title']} at {internship['company']}")
            st.write(f"Location: {internship['location']} | Duration: {internship['duration']} | Stipend: Rs {internship['stipend']}/month")
            st.write(f"Skills: {internship['skills']}")
        with col2:
            st.metric("Match Score", round(score, 3))
        st.markdown("---")

    st.markdown("### AI Career Advice:")
    with st.spinner("Generating advice..."):
        st.info(get_career_advice(query, context))

st.caption("Built with Endee Vector DB + Sentence Transformers + Groq AI")
