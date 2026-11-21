import os, glob, re
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="Zyro HR Help Desk", page_icon="💼")

# Keys come from Streamlit secrets (never hard-code them)
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
if "LANGCHAIN_API_KEY" in st.secrets:
    os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "zyro-rag-challenge"

REFUSAL = ("I'm sorry, but I can only answer questions about Zyro Dynamics HR policies, "
           "and this information is not available in the HR documents. "
           "Please contact the HR team directly for help.")

PROMPT = ChatPromptTemplate.from_template("""You are the Zyro Dynamics HR Help Desk assistant.
Answer ONLY from the context below. No outside knowledge.
Write a direct answer in plain text: no markdown, no bullet points, no bold, no headings.
Use 1 to 3 sentences. State the exact numbers, days, percentages, eligibility and conditions
as written in the policy, and nothing extra.
For timeline or process questions, list EVERY stage and its dates found in the context, in order,
in up to 4 sentences.
Words like "I", "my" and "me" mean the employee asking, so answer with the general policy rule.
Do not add any remark about personal or individual records.
If the context contains policy information that answers the question, even partly, answer with it.
Reply exactly with: "{refusal}"
only when the context has nothing relevant, or the question is about another company,
company financials, products, job applications or any non-HR topic.

Context:
{context}

Question: {question}
Answer:""")

ELIG_RULE = ("\n\nThis is an eligibility question. List EVERY eligibility condition "
             "stated in the context (service period, grade, performance rating, PIP or "
             "disciplinary status, role suitability, equipment or workspace), in plain text, "
             "in up to 4 sentences.")

PERSONAL = re.compile(r"\b(my|me)\b.*\b(schedule|balance|grant|payslip|vesting)\b", re.I)
PERSONAL_NOTE = " Individual details are not available in the HR documents."


def title_of(src):
    return re.sub(r"^\d+_", "", src).replace(".pdf", "").replace("_", " ")


@st.cache_resource(show_spinner="Loading HR policies...")
def build_index():
    docs = []
    for p in sorted(glob.glob("data/*.pdf")):
        for d in PyPDFLoader(p).load():
            d.metadata["source"] = os.path.basename(p)
            docs.append(d)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1400, chunk_overlap=250)
    chunks = [Document(page_content=f"{title_of(c.metadata['source'])}: {c.page_content}",
                       metadata=c.metadata)
              for c in splitter.split_documents(docs)]
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = FAISS.from_documents(chunks, emb)
    return vs.as_retriever(search_type="similarity", search_kwargs={"k": 6})


retriever = build_index()
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
chain = PROMPT | llm | StrOutputParser()


def ask_bot(question):
    ds = retriever.invoke(question)
    ctx = "\n\n".join(f"[{d.metadata['source']}] {d.page_content}" for d in ds)
    q_llm = question + (ELIG_RULE if re.search(r"eligib", question, re.I) else "")
    ans = chain.invoke({"context": ctx, "question": q_llm, "refusal": REFUSAL}).strip()
    low = ans.lower().replace("’", "'")
    refused = False
    if "only answer questions about" in low or "not available in the hr documents. please contact" in low:
        ans, refused = REFUSAL, True
    elif PERSONAL.search(question):
        ans += PERSONAL_NOTE
    sources = [] if refused else sorted({(d.metadata["source"], d.metadata.get("page", 0) + 1) for d in ds})
    return ans, sources


st.title("💼 Zyro Dynamics HR Help Desk")
st.caption("Ask questions about Zyro Dynamics HR policies.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])
        if m.get("sources"):
            with st.expander("Sources"):
                for f, pg in m["sources"]:
                    st.write(f"{f}, page {pg}")

if q := st.chat_input("Ask an HR policy question..."):
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            ans, sources = ask_bot(q)
        st.write(ans)
        if sources:
            with st.expander("Sources"):
                for f, pg in sources:
                    st.write(f"{f}, page {pg}")
    st.session_state.messages.append({"role": "assistant", "content": ans, "sources": sources})