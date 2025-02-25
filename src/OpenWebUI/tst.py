import requests
import json
import time
import logging
import asyncio
import aiohttp
from typing import Union, Generator, Tuple, List, Dict, Set
from pydantic import BaseModel, Field
from dataclasses import dataclass
from asyncio import Semaphore

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def pop_system_message(messages: List[Dict]) -> Tuple[str, List[Dict]]:
    """Extract system message from messages list"""
    system_message = ""
    other_messages = []

    for message in messages:
        if message["role"] == "system":
            system_message = message["content"]
        else:
            other_messages.append(message)

    return system_message, other_messages


class Pipe:
    class Valves(BaseModel):
        # API Configuration
        OPENAI_KEY: str = Field(
            default="",
            description="API key for OpenAI/OpenRouter",
        )
        OPENAI_MODEL: str = Field(
            default="openai/gpt-4o",
            description="Model to use for text generation (e.g., openai/gpt-3.5-turbo, anthropic/claude-2)",
        )
        OPENAI_URL: str = Field(
            default="https://openrouter.ai/api/v1",
            description="API endpoint URL (e.g., https://openrouter.ai/api/v1, https://api.openai.com/v1)",
        )
        FIRECRAWL_KEY: str = Field(
            default="",
            description="API key for Firecrawl",
        )
        FIRECRAWL_URL: str = Field(
            default="https://api.firecrawl.dev", description="Firecrawl API endpoint"
        )

        # Research Parameters
        MAX_TOTAL_QUERIES: int = Field(
            default=50, description="Maximum number of search queries to perform"
        )
        MAX_RESEARCH_TIME: int = Field(
            default=15, description="Maximum research time in minutes"
        )
        MAX_COST: float = Field(default=2.0, description="Maximum cost in dollars")
        MIN_NEW_INFO_RATIO: float = Field(
            default=0.3,
            description="Minimum ratio of new information required to continue",
        )
        MAX_SIMILAR_QUERIES: int = Field(
            default=3, description="Maximum number of similar queries allowed"
        )

        # Model Parameters
        CONTEXT_SIZE: int = Field(
            default=128000, description="Maximum context size for the model"
        )
        CONCURRENCY_LIMIT: int = Field(
            default=2, description="Maximum number of concurrent API calls"
        )
        MIN_CHUNK_SIZE: int = Field(
            default=140, description="Minimum size for text chunks"
        )

    def __init__(self):
        self.valves = self.Valves()
        self.semaphore = Semaphore(self.valves.CONCURRENCY_LIMIT)
        self.conversation_history = {}
        logger.debug("DeepR Pipe instance initialized")

    def _get_headers(self):
        headers = {
            "Authorization": f"Bearer {self.valves.OPENAI_KEY}",
            "Content-Type": "application/json",
        }
        logger.debug("Headers generated")
        return headers

    def pipes(self):
        """Return available models"""
        return [
            {
                "id": self.valves.OPENAI_MODEL,
                "name": f"DeepR Research Assistant ({self.valves.OPENAI_MODEL})",
            }
        ]

    @dataclass
    class ResearchLimits:
        max_total_queries: int
        max_research_time: int
        max_cost: float
        min_new_info_ratio: float
        max_similar_queries: int

    class ResearchTracker:
        def __init__(self, limits):
            self.limits = limits
            self.start_time = time.time()
            self.total_queries = 0
            self.total_cost = 0.0
            self.query_history: Set[str] = set()
            self.learning_history: Set[str] = set()

        def should_continue(self) -> Tuple[bool, str]:
            if (time.time() - self.start_time) > (self.limits.max_research_time * 60):
                return False, "Research time limit reached"
            if self.total_queries >= self.limits.max_total_queries:
                return False, "Maximum number of queries reached"
            if self.total_cost >= self.limits.max_cost:
                return False, "Research budget exceeded"
            return True, ""

        def add_query(self, query: str) -> bool:
            similar_count = sum(
                1
                for q in self.query_history
                if self._calculate_similarity(query, q) > 0.8
            )
            if similar_count >= self.limits.max_similar_queries:
                return False
            self.query_history.add(query)
            self.total_queries += 1
            return True

        def add_learnings(self, new_learnings: List[str]) -> float:
            if not new_learnings:
                return 0.0
            new_learning_set = set(new_learnings)
            new_info = new_learning_set - self.learning_history
            new_info_ratio = len(new_info) / len(new_learnings)
            self.learning_history.update(new_learning_set)
            return new_info_ratio

        def _calculate_similarity(self, str1: str, str2: str) -> float:
            def get_bigrams(s):
                return set(s[i : i + 2].lower() for i in range(len(s) - 1))

            bigrams1 = get_bigrams(str1)
            bigrams2 = get_bigrams(str2)
            if not bigrams1 or not bigrams2:
                return 0.0
            intersection = len(bigrams1 & bigrams2)
            union = len(bigrams1 | bigrams2)
            return intersection / union if union > 0 else 0.0

    async def generate_text(
        self, prompt: str, system: str = "", stream: bool = False
    ) -> Union[str, Generator]:
        """Generate text using the configured model"""
        try:
            url = f"{self.valves.OPENAI_URL}/chat/completions"
            headers = self._get_headers()

            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            payload = {
                "model": self.valves.OPENAI_MODEL,
                "messages": messages,
                "stream": stream,
                "temperature": 0.7,
                "max_tokens": 4096,
            }

            if stream:
                return self.stream_response(url, headers, payload)
            else:
                return await self.non_stream_response(url, headers, payload)

        except Exception as e:
            logger.error(f"Error in generate_text: {e}")
            return f"Error: {e}"

    def stream_response(self, url, headers, payload):
        try:
            with requests.post(
                url, headers=headers, json=payload, stream=True, timeout=(3.05, 60)
            ) as response:
                if response.status_code != 200:
                    raise Exception(
                        f"HTTP Error {response.status_code}: {response.text}"
                    )

                for line in response.iter_lines():
                    if line:
                        line = line.decode("utf-8")
                        if line.startswith("data: "):
                            try:
                                data = json.loads(line[6:])
                                if "choices" in data and len(data["choices"]) > 0:
                                    choice = data["choices"][0]
                                    delta = choice.get("delta", {})
                                    content = delta.get("content", "")
                                    if content:
                                        yield content
                                time.sleep(0.01)
                            except json.JSONDecodeError:
                                logger.error(f"Failed to parse JSON: {line}")
                            except KeyError as e:
                                logger.error(f"Unexpected data structure: {e}")
        except Exception as e:
            logger.error(f"Error in stream_response: {e}")
            yield f"Error: {e}"

    async def non_stream_response(self, url, headers, payload):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, headers=headers, json=payload, timeout=60
                ) as response:
                    if response.status != 200:
                        raise Exception(
                            f"HTTP Error {response.status}: {await response.text()}"
                        )

                    res = await response.json()
                    return (
                        res["choices"][0]["message"]["content"]
                        if "choices" in res and res["choices"]
                        else ""
                    )
        except Exception as e:
            logger.error(f"Error in non_stream_response: {e}")
            return f"Error: {e}"

    async def search(self, query: str, timeout: int = 15000, limit: int = 5) -> Dict:
        """Perform web search using Firecrawl"""
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {self.valves.FIRECRAWL_KEY}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
            payload = {
                "query": query,
                "limit": limit,
                "timeout": timeout,
                "scrapeOptions": {"formats": ["markdown"]},
            }

            try:
                async with session.post(
                    f"{self.valves.FIRECRAWL_URL}/v1/search",
                    headers=headers,
                    json=payload,
                    timeout=timeout / 1000,
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        raise Exception(f"Search failed with status {response.status}")
            except Exception as e:
                return {"error": str(e)}

    async def deep_research(self, query: str, depth: int = 2, breadth: int = 4) -> Dict:
        """Perform deep research on a topic"""
        limits = self.ResearchLimits(
            max_total_queries=self.valves.MAX_TOTAL_QUERIES,
            max_research_time=self.valves.MAX_RESEARCH_TIME,
            max_cost=self.valves.MAX_COST,
            min_new_info_ratio=self.valves.MIN_NEW_INFO_RATIO,
            max_similar_queries=self.valves.MAX_SIMILAR_QUERIES,
        )
        tracker = self.ResearchTracker(limits)

        all_learnings = []
        all_urls = []

        should_continue, stop_reason = tracker.should_continue()
        if not should_continue:
            return {"error": stop_reason}

        if not tracker.add_query(query):
            return {"error": "Query too similar to previous queries"}

        try:
            search_results = await self.search(query)
            if "error" in search_results:
                return {"error": search_results["error"]}

            urls = [item["url"] for item in search_results.get("data", [])]
            all_urls.extend(urls)

            contents = [
                item.get("markdown", "")
                for item in search_results.get("data", [])
                if item.get("markdown")
            ]

            prompt = f"""Given the following contents from a search for the query: "{query}", 
            extract the key learnings. Be detailed and include specific facts, numbers, and dates when available.

            Contents: {' '.join(contents)}"""

            learnings = await self.generate_text(prompt)
            learnings_list = [l.strip() for l in learnings.split("\n") if l.strip()]
            all_learnings.extend(learnings_list)

            if depth > 1:
                for learning in learnings_list[:breadth]:
                    if not tracker.should_continue()[0]:
                        break

                    sub_results = await self.deep_research(
                        learning, depth=depth - 1, breadth=breadth // 2
                    )

                    if "error" not in sub_results:
                        all_learnings.extend(sub_results.get("learnings", []))
                        all_urls.extend(sub_results.get("urls", []))

            # Remove duplicates while preserving order
            all_learnings = list(dict.fromkeys(all_learnings))
            all_urls = list(dict.fromkeys(all_urls))

            return {"learnings": all_learnings, "urls": all_urls}

        except Exception as e:
            return {"error": str(e)}

    async def pipe(self, body: dict) -> Union[str, Dict, Generator]:
        """
        Handle chat interactions and research requests

        Input body format:
        {
            "messages": List[dict],  # List of message objects including system and history
            "stream": bool,          # Whether to stream the response
            "research": bool,        # Whether to perform deep research
            "depth": int,           # Research depth (optional, default: 2)
            "breadth": int,         # Research breadth (optional, default: 4)
        }
        """
        try:
            messages = body.get("messages", [])
            if not messages:
                return "No messages provided"

            # Extract system message and handle conversation history
            system_message, messages = pop_system_message(messages)

            # Get the latest user message
            user_message = messages[-1]["content"] if messages else ""

            # Store conversation context
            conversation_id = body.get("conversation_id", "default")
            if conversation_id not in self.conversation_history:
                self.conversation_history[conversation_id] = []


            # Perform deep research
            depth = int(body.get("depth", 2))
            breadth = int(body.get("breadth", 4))

            research_results = await self.deep_research(
                user_message, depth, breadth
            )

            if "error" in research_results:
                return {"error": research_results["error"]}

            # Generate research summary
            learnings = research_results["learnings"]
            urls = research_results["urls"]

            summary_prompt = f"""Based on the research results, provide a detailed summary of findings about: "{user_message}"

            Key Learnings:
            {chr(10).join(f'- {learning}' for learning in learnings)}"""

            if body.get("stream", False):
                return self.stream_response(
                    url=f"{self.valves.OPENAI_URL}/chat/completions",
                    headers=self._get_headers(),
                    payload=
                    {
                        "model": self.valves.OPENAI_MODEL,
                        "messages": [
                            {"role": "system", "content": system_message} if system_message else {},
                            {"role": "user", "content": summary_prompt},
                        ],
                        "stream": True,
                    },
                )
            else:
                summary = await self.generate_text(summary_prompt, system_message)
                # Add sources at the end of the report
                final_report = f"{summary}\n\n## Sources\n\n" + "\n".join(
                    f"- {url}" for url in urls
                )
                return final_report

        except Exception as e:
            logger.error(f"Error in pipe method: {e}")
            return {"error": f"Error processing request: {str(e)}"}


# Example usage
if __name__ == "__main__":
    async def test():
        pipe = Pipe()
        
        """ # Test regular chat
        print("\nTesting regular chat:")
        result = await pipe.pipe({
            "messages": [
                {"role": "system", "content": "You are a helpful research assistant."},
                {"role": "user", "content": "What can you tell me about quantum computing?"}
            ],
            "stream": False
        })
        print(f"Chat Response: {result}") """

        # Test deep research
        print("\nTesting deep research:")
        print(await pipe.pipe({
            "messages": [
                {"role": "system", "content": "You are a helpful research assistant."},
                {"role": "user", "content": "What are the latest developments in quantum computing?"}
            ],
            "research": True,
            "depth": 2,
            "breadth": 3,
            "stream": True
        }))
        # print("\nResearch Results:")
        # print(f"Summary: {research_result.get('summary', '')}")
        # print(f"Number of learnings: {len(research_result.get('learnings', []))}")
        # print(f"Number of sources: {len(research_result.get('urls', []))}")

    asyncio.run(test())