# InternMatch AI

AI-powered internship finder that matches students to relevant internships
based on their skills using semantic search - powered by Endee Vector DB.

## Deploy Link
[https://internmatch-ai.onrender.com/](https://internmatch-ai.onrender.com/)

## Problem Statement

Students struggle to find relevant internships because normal search is
keyword-based. If a student types "I know React and Node.js", normal search
won't find "Full Stack Developer Intern" unless exact words match.

InternMatch AI uses semantic similarity to understand the meaning behind
skills and match them to the right internships.

Note: Live demo uses lightweight search for deployment. 
For full Endee Vector DB experience, run locally using Docker.

## How It Works

Student describes their skills in natural language
        |
Sentence Transformer converts it to a vector embedding
        |
Endee Vector DB finds top 5 semantically similar internships
        |
Groq AI gives personalized career advice

## How Endee is Used

- Index named "internships" created in Endee with 384 dimensions and cosine similarity
- Each internship description converted to vector embedding using sentence-transformers
- Vectors stored in Endee using index.upsert()
- Student query embedded and matched against stored vectors using index.query()
- Returns top 5 most semantically similar internships with similarity scores

## Tech Stack

- Endee Vector DB - vector storage and semantic search
- Sentence Transformers all-MiniLM-L6-v2 - text embeddings
- Groq AI llama-3.1-8b-instant - personalized career advice
- Streamlit - web UI
- Python 3.13

## Setup Instructions

1. Clone this repository:
   git clone https://github.com/Mishra-coder/endee.git
   cd endee

2. Start Endee server using Docker:
   docker compose up -d

3. Create virtual environment:
   python3 -m venv venv
   source venv/bin/activate

4. Install dependencies:
   pip install endee sentence-transformers streamlit groq python-dotenv

5. Add your Groq API key:
   echo "GROQ_API_KEY=your_key_here" > .env

6. Run the app:
   streamlit run app.py

7. Open http://localhost:8501

## Usage

- Describe your skills and interests in the text box
- Click "Find Matching Internships"
- Get top 5 matching internships with similarity scores
- Get AI-powered personalized career advice

## Example Queries

- "I know Python, machine learning basics and data analysis"
- "I know React, Node.js, MongoDB and want a full stack role"
- "I have experience with Android development using Kotlin"
