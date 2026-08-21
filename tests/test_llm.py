from app.generation.llm import LLMGenerator


llm = LLMGenerator()


response = llm.generate(
    query="What is the purpose of this test?",
    context="""
    This is a test document.
    Its purpose is to verify that the Gemini API
    connection is working correctly.
    """
)


print("\n" + "=" * 80)
print("LLM RESPONSE")
print("=" * 80)

print(response)