import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors


load_dotenv()


class LLMGenerator:
    """
    Handles LLM-based answer generation using Gemini.
    """

    def __init__(
        self,
        model: str = "gemini-3.6-flash"
    ):

        api_key = os.getenv(
            "GEMINI_API_KEY"
        )

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. "
                "Please add it to your .env file."
            )

        self.client = genai.Client(
            api_key=api_key
        )

        self.model = model

    def generate(
        self,
        query: str,
        context: str
    ) -> str:

        if not context.strip():

            return (
                "I couldn't find sufficient information "
                "in the provided documents to answer "
                "this question."
            )

        prompt = f"""
You are an intelligent document analysis assistant.

Your task is to answer the user's question using
ONLY the information contained in the provided
document sources.

STRICT RULES:

1. Use ONLY the provided sources as evidence.

2. Do NOT use outside knowledge.

3. Do NOT make assumptions that are not supported
   by the sources.

4. Every factual statement in your answer MUST have
   a citation in the format [1], [2], [3], etc.

5. The citation number MUST correspond to the
   Source ID provided in the context.

6. If multiple sources support a statement, cite
   all relevant sources.

7. If the provided sources do not contain enough
   information to answer the question, say:

   "I couldn't find sufficient information in the
   provided documents to answer this question."

8. Do NOT attempt to answer using your general
   knowledge when the documents do not contain
   the answer.

9. Keep the answer concise but complete.

10. Do not invent source numbers.

DOCUMENT SOURCES:
==================================================

{context}

==================================================

USER QUESTION:

{query}

==================================================

ANSWER WITH CITATIONS:
"""

        # Retry temporary Gemini server errors
        for attempt in range(3):

            try:

                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt
                )

                return response.text

            except errors.ServerError:

                if attempt == 2:
                    raise

                time.sleep(2 ** attempt)