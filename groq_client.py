import os
import time
import logging
from groq import Groq, RateLimitError, APIError
from utils.rate_limiter import KeyRotator

logger = logging.getLogger("groq_client")


class MultiKeyGroqClient:
    """
    Enterprise-grade Groq client with:
    - Dynamic API key rotation across N keys
    - Per-key sliding-window rate limiting
    - Exponential backoff on RateLimitError
    - Automatic retry with key fallback on APIError
    """

    def __init__(self, api_keys: list, requests_per_minute: int = 25):
        if not api_keys:
            raise ValueError("At least one Groq API key must be provided.")
        self.rotator = KeyRotator(api_keys, requests_per_minute)
        self.max_retries = len(api_keys) * 2  # Allow each key at least 2 attempts

    def generate_report(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: dict = None,
    ) -> str:
        """
        Generates a completion with retry + key-rotation logic.
        Passing response_format={"type":"json_object"} enables structured JSON output.
        """
        last_error = None
        for attempt in range(self.max_retries):
            api_key, limiter = self.rotator.get_next_key_info()
            limiter.wait_if_needed()

            try:
                client = Groq(api_key=api_key)

                # Build kwargs conditionally — passing response_format=None
                # causes a Groq API validation error, so only include when set.
                kwargs = dict(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.2,
                    max_tokens=1024,
                )
                if response_format:
                    kwargs["response_format"] = response_format

                response = client.chat.completions.create(**kwargs)
                return response.choices[0].message.content

            except RateLimitError as e:
                last_error = e
                logger.warning(
                    f"Rate limit hit on key #{self.rotator.current_index} "
                    f"(attempt {attempt+1}/{self.max_retries}). Rotating & backing off..."
                )
                backoff = min(30, (2 ** attempt) + (time.time() % 1))
                time.sleep(backoff)

            except APIError as e:
                last_error = e
                logger.warning(
                    f"Groq APIError on attempt {attempt+1}: {e}. "
                    f"Rotating key and retrying..."
                )
                if attempt == self.max_retries - 1:
                    raise RuntimeError(
                        f"Groq API failed after {self.max_retries} attempts. "
                        f"Last error: {e}"
                    ) from e
                time.sleep(1 + attempt)

        raise RuntimeError(
            f"Groq API exhausted all {self.max_retries} retries. "
            f"Last error: {last_error}"
        )

    def chat_query(self, context: str, query: str) -> str:
        """
        RAG-lite chat: grounds responses strictly in the provided news context.
        No response_format enforced here — free-text analyst reply.
        """
        system_prompt = (
            "You are a Senior Financial Analyst at an institutional hedge fund. "
            "Answer the user's question using ONLY the provided news article context below. "
            "If the answer cannot be derived from the context, clearly state: "
            "'I do not have sufficient context to answer this question.' "
            "Never hallucinate. Be concise, professional, and data-driven."
        )
        user_prompt = (
            f"News Context:\n{context}\n\n"
            f"---\n"
            f"Analyst Question: {query}"
        )
        return self.generate_report(system_prompt, user_prompt)
