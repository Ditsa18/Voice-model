import time

from abp_voice.rag.pipeline import RAGPipeline

pipeline = RAGPipeline()

print("Warming up...")
pipeline.warmup()

while True:

    question = input("\nYou: ")

    if question.lower() in ["exit", "quit"]:
        break

    start = time.time()

    result = pipeline.generate(
        question=question,
        lang="en"
    )

    total = time.time() - start

    print("Assistant:", result.answer)
    print(f"Latency: {total:.2f}s")