import os
import streamlit as st
import spacy
import networkx as nx
import string, re
from collections import Counter
from dotenv import load_dotenv
from fuzzywuzzy import fuzz
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load environment variables
load_dotenv()
nlp = spacy.load("en_core_web_sm")

# Utility functions
STOP_WORDS = spacy.lang.en.stop_words.STOP_WORDS
punct_table = str.maketrans("", "", string.punctuation)

def is_valid_node(text: str) -> bool:
    txt = text.lower().translate(punct_table).strip()
    if len(txt) < 4:
        return False
    if txt in STOP_WORDS:
        return False
    if re.fullmatch(r"\d+", txt):
        return False
    BAD = {"another", "other", "type", "types", "whichever", "availability"}
    if txt in BAD:
        return False
    return True

def extract_triples(text: str):
    doc = nlp(text)
    triples = []

    for sent in doc.sents:
        added = False
        for verb in [t for t in sent if t.pos_ == "VERB"]:
            subj = [c for c in verb.children if c.dep_ in ("nsubj", "nsubjpass")]
            obj = [c for c in verb.children if c.dep_ in ("dobj", "pobj", "attr", "obl")]
            if subj and obj:
                h = subj[0].text.strip()
                t = obj[0].text.strip()
                if is_valid_node(h) and is_valid_node(t):
                    triples.append((h, verb.lemma_, t))
                    added = True

        noun_chunks = [chunk.text.strip() for chunk in sent.noun_chunks if is_valid_node(chunk.text.strip())]
        for i in range(len(noun_chunks)):
            for j in range(i + 1, len(noun_chunks)):
                triples.append((noun_chunks[i], "related_to", noun_chunks[j]))

    return triples

# Streamlit UI
st.title("📚 Graph-RAG over PDFs with Groq + LLaMA3")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    st.success("PDF uploaded. Processing...")

    os.makedirs("documents", exist_ok=True)
    file_path = f"documents/{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    loader = PyPDFLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    st.write(f"Total Chunks: {len(chunks)}")

    if not chunks:
        st.error("No valid text found in the PDF. Please upload a different file.")
        st.stop()

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    G = nx.DiGraph()
    all_triples = []
    edge_counter = Counter()

    for chunk in chunks:
        triples = extract_triples(chunk.page_content)
        all_triples.extend(triples)
        edge_counter.update(triples)

    for h, r, t in all_triples:
        if edge_counter[(h, r, t)] >= 1:
            G.add_edge(h, t, relation=r)

    st.success("Document processed and knowledge graph built.")

    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model="Llama3-8b-8192"
    )

    query = st.text_input("Ask a question based on the PDF + Graph:")

    if query:
        def graph_retrieve(query):
            doc = nlp(query)
            query_terms = set(ent.text for ent in doc.ents).union(
                set(chunk.text.strip() for chunk in doc.noun_chunks if is_valid_node(chunk.text.strip()))
            )
            results = []
            for node in G.nodes:
                for term in query_terms:
                    score = fuzz.partial_ratio(term.lower(), node.lower())
                    if score > 80:
                        for neighbor in G.neighbors(node):
                            rel = G.get_edge_data(node, neighbor).get("relation", "related_to")
                            fact = f"{node} --[{rel}]--> {neighbor}"
                            results.append((score, fact))
            results.sort(reverse=True)
            return "\n".join([fact for _, fact in results[:10]]) if results else "No graph data found."

        graph_context = graph_retrieve(query)
        vector_docs = vectorstore.similarity_search(query, k=3)
        vector_context = "\n".join([doc.page_content for doc in vector_docs])
        full_context = vector_context + "\n" + graph_context

        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
You are a smart assistant. Use the following context (from documents and knowledge graph) to answer the question.

Context:
{context}

Question: {question}

Answer:
"""
        )

        chain = LLMChain(llm=llm, prompt=prompt)
        answer = chain.run({"context": full_context, "question": query})

        st.markdown("### ✅ Answer:")
        st.success(answer)

        with st.expander("\U0001F4CA Graph Facts Used"):
            st.code(graph_context)
