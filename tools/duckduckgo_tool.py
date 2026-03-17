from langchain_community.tools import DuckDuckGoSearchRun
from transformers import pipeline

search_tool = DuckDuckGoSearchRun()
results = search_tool.invoke("top news in india today")

print("Raw Results:\n", results)

generator = pipeline(
    "text-generation",
    model="gpt2"  
) 

prompt = f"Summarize this news in short:\n{results}\nSummary:"

output = generator(
    prompt,
    max_length=150,
    num_return_sequences=1
)

print("\nSummary:\n", output[0]['generated_text'])