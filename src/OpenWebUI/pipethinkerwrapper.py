import requests
import json
import time
import logging
import asyncio
import aiohttp
from typing import Union, Generator, Tuple, List, Dict, Set
from pydantic import BaseModel, Field
from dataclasses import dataclass
from collections import deque

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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
            description="API key for OpenAI/OpenRouter"
        )
        OPENAI_MODEL: str = Field(
            default="deepseek-ai/DeepSeek-R1",
            description="Model to use for text generation (e.g., openai/gpt-3.5-turbo, anthropic/claude-2)"
        )
        OPENAI_URL: str = Field(
            default="https://openrouter.ai/api/v1",
            description="API endpoint URL (e.g., https://openrouter.ai/api/v1, https://api.openai.com/v1)"
        )
        
        MAX_COST: float = Field(
            default=2.0,
            description="Maximum cost in dollars"
        )
        
        # Model Parameters
        CONTEXT_SIZE: int = Field(
            default=128000,
            description="Maximum context size for the model"
        )
        CONCURRENCY_LIMIT: int = Field(
            default=2,
            description="Maximum number of concurrent API calls"
        )
        MIN_CHUNK_SIZE: int = Field(
            default=140,
            description="Minimum size for text chunks"
        )

    def __init__(self):
        self.valves = self.Valves()
        self.conversation_history = {}
        logger.debug("R1 Thinker Wrapper")

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
                "name": f"Thinker Wrapper ({self.valves.OPENAI_MODEL})"
            }
        ]

    @dataclass
    class Limits:
        max_cost: float

    class Tracker:
        def __init__(self, limits):
            self.limits = limits
            self.start_time = time.time()
            self.query_history: Set[str] = set()

            
        def should_continue(self) -> Tuple[bool, str]:
            if self.total_cost >= self.limits.max_cost:
                return False, "Budget exceeded"
            return True, ""

    async def generate_text(self, prompt: str, system: str = "", stream: bool = True) -> Union[str, Generator]:
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
                "include_reasoning": True
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
                    raise Exception(f"HTTP Error {response.status_code}: {response.text}")
                
                in_reasoning = True
                content_buffer = deque()
                consecutive_empty_reasoning = 0
                MAX_EMPTY_REASONING = 10  # Number

                for line in response.iter_lines():
                    if line:
                        line = line.decode("utf-8")
                        if line.startswith("data: "):
                            try:
                                data = json.loads(line[6:])
                                if "choices" in data and len(data["choices"]) > 0:
                                    content = data["choices"][0].get("delta", {}).get("content", "")
                                    reasoning = data["choices"][0].get("delta", {}).get("reasoning", "")

                                    # Handle reasoning phase
                                    if in_reasoning:
                                        if reasoning:
                                            yield reasoning
                                            consecutive_empty_reasoning = 0
                                        else:
                                            consecutive_empty_reasoning += 1
                                        
                                        # Check for transition conditions
                                        if consecutive_empty_reasoning >= MAX_EMPTY_REASONING:
                                            in_reasoning = False
                                            yield "\n\n---\n\n"  # Add separator after reasoning
                                            # Flush content buffer
                                            while content_buffer:
                                                buffered_content = content_buffer.popleft()
                                                if buffered_content.strip():
                                                    yield buffered_content
                                    
                                    # Handle content streaming
                                    if content:
                                        if in_reasoning:
                                            content_buffer.append(content)
                                        else:
                                            if content.strip():
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
                async with session.post(url, headers=headers, json=payload, timeout=60) as response:
                    if response.status != 200:
                        raise Exception(f"HTTP Error {response.status}: {await response.text()}")
                    
                    res = await response.json()
                    return (
                        res["choices"][0]["message"]["content"]
                        if "choices" in res and res["choices"]
                        else ""
                    )
        except Exception as e:
            logger.error(f"Error in non_stream_response: {e}")
            return f"Error: {e}"

    async def pipe(self, body: dict) -> Union[str, Dict, Generator]:
        """
        Handle chat interactions and research requests
        
        Input body format:
        {
            "messages": List[dict],  # List of message objects including system and history
            "stream": bool,          # Whether to stream the response
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
            
            # Regular chat completion
            if body.get("stream", True):
                return self.stream_response(
                    f"{self.valves.OPENAI_URL}/chat/completions",
                    self._get_headers(),
                    {
                        "model": self.valves.OPENAI_MODEL,
                        "messages": [
                            {"role": "system", "content": system_message} if system_message else {},
                            *messages
                        ],
                        "stream": True,
                        "include_reasoning": True
                    }
                )
            else:
                return await self.generate_text(user_message, system_message)

        except Exception as e:
            logger.error(f"Error in pipe method: {e}")
            return {"error": f"Error processing request: {str(e)}"}

# Example usage
if __name__ == "__main__":
    async def test():
        pipe = Pipe()
        
        # Test regular chat
        print("\nTesting regular chat:")
        result = await pipe.pipe({
            "messages": [
                {"role": "system", "content": "You are a helpful research assistant."},
                {"role": "user", "content": "What can you tell me about quantum computing?"}
            ],
            "stream": True
        })
        print(f"Chat Response: {result}")

        """ # Test deep research
        print("\nTesting deep research:")
        research_result = await pipe.pipe({
            "messages": [
                {"role": "system", "content": "You are a helpful research assistant."},
                {"role": "user", "content": "What are the latest developments in quantum computing?"}
            ],
            "research": True,
            "depth": 2,
            "breadth": 3,
            "stream": False
        })
        print("\nResearch Results:")
        print(f"Summary: {research_result.get('summary', '')}")
        print(f"Number of learnings: {len(research_result.get('learnings', []))}")
        print(f"Number of sources: {len(research_result.get('urls', []))}") """

    asyncio.run(test())