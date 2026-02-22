from retrieval import make_multi_query_rewriter

rewriter = make_multi_query_rewriter(model="mistral", n=3)
question = "is there any discussion about chatbot"
queries = rewriter.invoke(question)

print("--- REWRITTEN QUERIES ---")
for i, q in enumerate(queries):
    print(f"{i}: {q}")
print("-------------------------")
