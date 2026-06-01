import time

from abp_voice.rag.pipeline import RAGPipeline

pipeline = RAGPipeline()

print("warming up...")
pipeline.warmup()

print("warmup complete")

start = time.time()

result = pipeline.generate(
    "What is ABP?",
    "en"
)

print(result.answer)

print(
    "latency:",
    round(time.time() - start, 2),
    "seconds"
)