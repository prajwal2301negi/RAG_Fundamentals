
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings

video_id = "Gfr50f6ZBvo" 

try:
    transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=["en"]).to_raw_data()
    transcript = " ".join(chunk["text"] for chunk in transcript_list)
    print(transcript)
except TranscriptsDisabled:
    print("No captions available for this video.")
    transcript = ""


splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2" 
)

# Create FAISS vectorstore
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})


llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    temperature=0,
    max_new_tokens=512
)


prompt = PromptTemplate(
    template="""
You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, just say you don't know.

Transcript context:
{context}

Question:
{question}
""",
    input_variables=['context', 'question']
)

# Helper function to format docs
def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)


# Parallel chain: retrieve docs + question
parallel_chain = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

parser = StrOutputParser()
main_chain = parallel_chain | prompt | llm | parser


question1 = "Is the topic of nuclear fusion discussed in this video? If yes, what was discussed?"
question2 = "Can you summarize the video?"

answer1 = main_chain.invoke(question1)
answer2 = main_chain.invoke(question2)

print("\n--- Answer 1 ---")
print(answer1)
print("\n--- Answer 2 ---")
print(answer2)

