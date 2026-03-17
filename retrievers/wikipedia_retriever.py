from langchain_community.retrievers import WikipediaRetriever
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# HuggingFace model
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    temperature=0
)

model = ChatHuggingFace(llm=llm)

# Initialize Wikipedia retriever
retriever = WikipediaRetriever(
    top_k_results=2,
    lang="en"
)

query = "the geopolitical history of india and pakistan from the perspective of a chinese"

# Retrieve documents
docs = retriever.invoke(query)

# Combine retrieved text
context = "\n\n".join([doc.page_content for doc in docs])

# Prompt
prompt = PromptTemplate(
    template="""
Answer the following question using the given context.

Question:
{question}

Context:
{context}
""",
    input_variables=["question", "context"]
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({
    "question": query,
    "context": context
})

print("\nFinal Answer:\n")
print(result)

# Optional: print retrieved documents
for i, doc in enumerate(docs):
    print(f"\n--- Retrieved Document {i+1} ---")
    print(doc.page_content[:500])