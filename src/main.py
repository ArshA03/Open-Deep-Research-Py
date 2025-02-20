import os
from dotenv import load_dotenv
import sys
import time
import asyncio
import aiohttp
import tiktoken
import datetime
from openai import AsyncOpenAI
from typing import List, Dict, Optional,Tuple, Set
from dataclasses import dataclass
import re
from asyncio import Semaphore
from helpers.logger import LogManager

# Initialize logging
logger = LogManager()

# Load environment variables from .env.local
load_dotenv('.env.local')

# Environment variables
FIRECRAWL_KEY = os.getenv("FIRECRAWL_KEY", "")
OPENAI_KEY = os.getenv("OPENAI_KEY", "")
OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "")
CONTEXT_SIZE = int(os.getenv("CONTEXT_SIZE", "128000"))

print("key is:", OPENAI_KEY)

# Research Limits
MAX_TOTAL_QUERIES = int(os.getenv("MAX_QUERIES", "50"))
MAX_RESEARCH_TIME = int(os.getenv("MAX_RESEARCH_TIME", "15"))  # 30 minutes
MAX_COST = float(os.getenv("MAX_COST", "2"))  # dollars
MIN_NEW_INFO_RATIO = float(os.getenv("MIN_NEW_INFO_RATIO", "0.3"))
MAX_SIMILAR_QUERIES = int(os.getenv("MAX_SIMILAR_QUERIES", "3"))

# Constants
CONCURRENCY_LIMIT = 2
MIN_CHUNK_SIZE = 140

@dataclass
class ResearchLimits:
    max_total_queries: int = MAX_TOTAL_QUERIES
    max_research_time: int = MAX_RESEARCH_TIME
    max_cost: float = MAX_COST
    min_new_info_ratio: float = MIN_NEW_INFO_RATIO
    max_similar_queries: int = MAX_SIMILAR_QUERIES

class ResearchTracker:
    def __init__(self, limits: ResearchLimits):
        self.limits = limits
        self.start_time = time.time()
        self.total_queries = 0
        self.total_cost = 0.0
        self.query_history: Set[str] = set()
        self.learning_history: Set[str] = set()
        
    def should_continue(self) -> Tuple[bool, str]:
        # Time limit check
        if (time.time() - self.start_time) > self.limits.max_research_time:
            return False, "Research time limit reached"
            
        # Query limit check
        if self.total_queries >= self.limits.max_total_queries:
            return False, "Maximum number of queries reached"
            
        # Cost limit check
        if self.total_cost >= self.limits.max_cost:
            return False, "Research budget exceeded"
            
        return True, ""

    def add_query(self, query: str) -> bool:
        # Check for similar queries using basic string similarity
        similar_count = sum(1 for q in self.query_history
                          if self._calculate_similarity(query, q) > 0.8)
        
        if similar_count >= self.limits.max_similar_queries:
            return False
            
        self.query_history.add(query)
        self.total_queries += 1
        return True

    def add_learnings(self, new_learnings: List[str]) -> float:
        if not new_learnings:
            return 0.0
            
        # Convert new learnings to set for comparison
        new_learning_set = set(new_learnings)
        new_info = new_learning_set - self.learning_history
        new_info_ratio = len(new_info) / len(new_learnings)
        
        # Update learning history
        self.learning_history.update(new_learning_set)
        return new_info_ratio

    def _calculate_similarity(self, str1: str, str2: str) -> float:
        # Simple similarity calculation using character bigrams
        def get_bigrams(s):
            return set(s[i:i+2].lower() for i in range(len(s)-1))
            
        bigrams1 = get_bigrams(str1)
        bigrams2 = get_bigrams(str2)
        
        if not bigrams1 or not bigrams2:
            return 0.0
            
        intersection = len(bigrams1 & bigrams2)
        union = len(bigrams1 | bigrams2)
        
        return intersection / union if union > 0 else 0.0

@dataclass
class ResearchProgress:
    current_depth: int
    total_depth: int
    current_breadth: int
    total_breadth: int
    current_query: Optional[str]
    total_queries: int
    completed_queries: int

@dataclass
class FeedbackQA:
    question: str
    answer: str

@dataclass
class SerpQuery:
    query: str
    research_goal: str

class OutputManager:
    def __init__(self):
        self.progress_lines = 4
        self.progress_area = []
        self.initialized = False
        sys.stdout.write('\n' * self.progress_lines)
        self.initialized = True

    def log(self, *args):
        if self.initialized:
            sys.stdout.write(f"\x1B[{self.progress_lines}A")
            sys.stdout.write("\x1B[0J")
        print(*args)
        if self.initialized:
            self.draw_progress()

    def update_progress(self, progress: ResearchProgress):
        self.progress_area = [
            f"Depth:    [{self.get_progress_bar(progress.total_depth - progress.current_depth, progress.total_depth)}] {round((progress.total_depth - progress.current_depth) / progress.total_depth * 100)}%",
            f"Breadth:  [{self.get_progress_bar(progress.total_breadth - progress.current_breadth, progress.total_breadth)}] {round((progress.total_breadth - progress.current_breadth) / progress.total_breadth * 100)}%",
            f"Queries:  [{self.get_progress_bar(progress.completed_queries, progress.total_queries)}] {round(progress.completed_queries / progress.total_queries * 100)}%",
            f"Current:  {progress.current_query}" if progress.current_query else ''
        ]
        self.draw_progress()

    def get_progress_bar(self, value: int, total: int) -> str:
        width = min(30, os.get_terminal_size().columns - 20)
        filled = round((width * value) / total)
        return '█' * filled + ' ' * (width - filled)

    def draw_progress(self):
        if not self.initialized or not self.progress_area:
            return
        terminal_height = os.get_terminal_size().lines
        sys.stdout.write(f"\x1B[{terminal_height - self.progress_lines};1H")
        sys.stdout.write('\n'.join(self.progress_area))
        sys.stdout.write(f"\x1B[{terminal_height - self.progress_lines - 1};1H")

class TextSplitter:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("Cannot have chunk_overlap >= chunk_size")
        self.separators = ['\n\n', '\n', '.', ',', '>', '<', ' ', '']

    def split_text(self, text: str) -> List[str]:
        final_chunks = []
        separator = self.separators[-1]
        
        for s in self.separators:
            if s == '':
                separator = s
                break
            if s in text:
                separator = s
                break

        splits = text.split(separator) if separator else list(text)
        good_splits = []

        for s in splits:
            if len(s) < self.chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    merged_text = self.merge_splits(good_splits, separator)
                    final_chunks.extend(merged_text)
                    good_splits = []
                other_info = self.split_text(s)
                final_chunks.extend(other_info)

        if good_splits:
            merged_text = self.merge_splits(good_splits, separator)
            final_chunks.extend(merged_text)

        return final_chunks

    def merge_splits(self, splits: List[str], separator: str) -> List[str]:
        docs = []
        current_doc = []
        total = 0

        for d in splits:
            _len = len(d)
            if total + _len >= self.chunk_size:
                if total > self.chunk_size:
                    logger.log_warning(f"Created a chunk of size {total}, which is longer than the specified {self.chunk_size}")
                if current_doc:
                    doc = self.join_docs(current_doc, separator)
                    if doc is not None:
                        docs.append(doc)
                    while (total > self.chunk_overlap or 
                           (total + _len > self.chunk_size and total > 0)):
                        total -= len(current_doc[0])
                        current_doc.pop(0)
            current_doc.append(d)
            total += _len

        doc = self.join_docs(current_doc, separator)
        if doc is not None:
            docs.append(doc)
        return docs

    def join_docs(self, docs: List[str], separator: str) -> Optional[str]:
        text = separator.join(docs).strip()
        return text if text else None

def trim_prompt(prompt: str, context_size: int = CONTEXT_SIZE) -> str:
    if not prompt:
        return ""
    
    encoder = tiktoken.get_encoding("o200k_base")  # Changed to match TypeScript version
    length = len(encoder.encode(prompt))
    
    if length <= context_size:
        return prompt

    overflow_tokens = length - context_size
    chunk_size = len(prompt) - overflow_tokens * 3

    if chunk_size < MIN_CHUNK_SIZE:
        return prompt[:MIN_CHUNK_SIZE]

    splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    trimmed_prompt = splitter.split_text(prompt)[0] or ""

    if len(trimmed_prompt) == len(prompt):
        return trim_prompt(prompt[:chunk_size], context_size)

    return trim_prompt(trimmed_prompt, context_size)

class OpenAIClient:
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=OPENAI_KEY,
            base_url=OPENAI_ENDPOINT if OPENAI_ENDPOINT else None
        )

    async def generate_text(self, prompt: str, system: str = "") -> str:
        try:
            start_time = time.time()
            logger.log_info(f"Generating text with {OPENAI_MODEL}")
            logger.log_prompt(system=system, user=prompt, model=OPENAI_MODEL, response="", tokens=None)
            
            response = await self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt}
                ],
                timeout=60
            )
            
            duration = time.time() - start_time
            result = response.choices[0].message.content
            
            # Log the complete interaction
            logger.log_prompt(
                system=system,
                user=prompt,
                response=result,
                model=OPENAI_MODEL,
                tokens=response.usage.total_tokens if response.usage else None
            )
            logger.log_info(f"Text generation completed in {duration:.2f}s")
            
            return result
        except Exception as e:
            logger.log_error(e, "Error generating text from OpenAI")
            return ""

class FirecrawlClient:
    def __init__(self):
        self.api_key = FIRECRAWL_KEY
        self.base_url = os.getenv("FIRECRAWL_BASE_URL", "https://api.firecrawl.dev")

    async def search(self, query: str, timeout: int = 15000, limit: int = 5) -> Dict:
        logger.log_info(f"Starting Firecrawl search for: {query}")
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
                "Content-Type": "application/json"
            }
            payload = {
                "query": query,
                "limit": limit,
                "timeout": timeout,
                "scrapeOptions": {"formats": ["markdown"]}
            }
            
            try:
                start_time = time.time()
                logger.log_debug(f"Sending search request to Firecrawl:\nQuery: {query}\nLimit: {limit}\nTimeout: {timeout}")
                
                async with session.post(
                    f"{self.base_url}/v1/search",
                    headers=headers,
                    json=payload,
                    timeout=timeout/1000
                ) as response:
                    duration = time.time() - start_time
                    if response.status == 200:
                        result = await response.json()
                        
                        # Log the API call details
                        logger.log_api_call(
                            endpoint=f"{self.base_url}/v1/search",
                            method="POST",
                            params={"query": query, "limit": limit, "timeout": timeout},
                            response=result,
                            duration=duration
                        )
                        
                        # Log search results
                        if result.get('data'):
                            logger.log_search_result(
                                query=query,
                                num_results=len(result['data']),
                                urls=[item.get('url') for item in result['data'] if item.get('url')]
                            )
                            logger.log_info(f"Search completed in {duration:.2f}s, found {len(result['data'])} results")
                        
                        return result
                    else:
                        error_msg = f"Search failed with status {response.status}"
                        logger.log_error(Exception(error_msg), f"Firecrawl search error for query: {query}")
                        raise Exception(error_msg)
            except asyncio.TimeoutError as e:
                error_msg = f"Timeout error for query: {query}"
                logger.log_error(e, error_msg)
                raise Exception(error_msg)
            except Exception as e:
                logger.log_error(e, f"Unexpected error in Firecrawl search for query: {query}")
                raise e

class NoSuchToolError(Exception):
    @staticmethod
    def isInstance(error):
        return isinstance(error, NoSuchToolError)

class InvalidToolArgumentsError(Exception):
    @staticmethod
    def isInstance(error):
        return isinstance(error, InvalidToolArgumentsError)

class ToolExecutionError(Exception):
    @staticmethod
    def isInstance(error):
        return isinstance(error, ToolExecutionError)

def system_prompt() -> str:
    now = datetime.datetime.now().isoformat()
    return f"""You are an expert researcher. Today is {now}. Follow these instructions when responding:
        - You may be asked to research subjects that is after your knowledge cutoff, assume the user is right when presented with news.
        - The user is a highly experienced analyst, no need to simplify it, be as detailed as possible and make sure your response is correct.
        - Be highly organized.
        - Suggest solutions that I didn't think about.
        - Be proactive and anticipate my needs.
        - Treat me as an expert in all subject matter.
        - Mistakes erode my trust, so be accurate and thorough.
        - Provide detailed explanations, I'm comfortable with lots of detail.
        - Value good arguments over authorities, the source is irrelevant.
        - Consider new technologies and contrarian ideas, not just the conventional wisdom.
        - You may use high levels of speculation or prediction, just flag it for me."""

# Initialize global instances
openai_client = OpenAIClient()
firecrawl_client = FirecrawlClient()
output_manager = OutputManager()
semaphore = Semaphore(CONCURRENCY_LIMIT)

def log(*args):
    """Log to both output manager and logger"""
    message = ' '.join(str(arg) for arg in args)
    output_manager.log(*args)
    logger.log_info(message)

async def generate_feedback(query: str, num_questions: int = 3) -> List[str]:
    """
    Generates feedback questions based on a user query
    
    Args:
        query: The user's research query
        num_questions: Maximum number of questions to generate (default: 3)
    
    Returns:
        List of follow-up questions
    """
    logger.log_info(f"Generating feedback questions for query: {query}")
    try:
        prompt = f"""Given the following query from the user, ask some follow up questions to clarify the research direction.
        The questions should help understand:
        1. The specific aspects or areas the user wants to focus on
        2. Any time periods or geographical regions of interest
        3. The level of technical detail required
        4. Any specific perspectives or approaches to consider
        
        Return a maximum of {num_questions} questions, but feel free to return less if the original query is clear.
        Format each question on a new line starting with a number.
        
        Query: <query>{query}</query>"""
        
        result = await openai_client.generate_text(prompt, system_prompt())
        
        # Extract questions from the result
        questions = []
        for line in result.split('\n'):
            line = line.strip()
            # Match lines that start with a number followed by a dot or parenthesis
            if re.match(r'^\d+[\.\)]', line):
                # Remove the number and any leading/trailing whitespace
                question = re.sub(r'^\d+[\.\)]\s*', '', line).strip()
                if question:
                    questions.append(question)
        
        questions = questions[:num_questions]
        logger.log_info(f"Generated {len(questions)} feedback questions")
        logger.log_debug("Feedback questions:\n" + "\n".join(f"  {i+1}. {q}" for i, q in enumerate(questions)))
        return questions
        
    except Exception as error:
        error_msg = ""
        if isinstance(error, NoSuchToolError):
            error_msg = f"Could not generate questions due to missing tool: {str(error)}"
            logger.log_error(error, "Tool not found during feedback generation")
        elif isinstance(error, InvalidToolArgumentsError):
            error_msg = f"Could not generate questions due to invalid arguments: {str(error)}"
            logger.log_error(error, "Invalid arguments during feedback generation")
        elif isinstance(error, ToolExecutionError):
            error_msg = f"Could not generate questions due to execution error: {str(error)}"
            logger.log_error(error, "Tool execution error during feedback generation")
        else:
            error_msg = "Could not generate questions due to an unexpected error"
            logger.log_error(error, "Unexpected error during feedback generation")
        return [error_msg]

async def collect_feedback(questions: List[str]) -> List[FeedbackQA]:
    """
    Collects user answers for each feedback question
    
    Args:
        questions: List of feedback questions to ask
    
    Returns:
        List of FeedbackQA pairs containing questions and their answers
    """
    logger.log_info(f"Starting feedback collection for {len(questions)} questions")
    feedback = []
    if not questions:
        logger.log_warning("No feedback questions provided")
        return feedback
        
    print("\nTo help focus the research, please answer these questions:")
    print("(Press Enter to skip any question)\n")
    
    for i, question in enumerate(questions, 1):
        answer = input(f"{i}. {question}\nYour answer: ").strip()
        if answer:  # Only include questions that received an answer
            feedback.append(FeedbackQA(question=question, answer=answer))
            logger.log_debug(f"Collected answer for question {i}: {question}")
        else:
            logger.log_debug(f"Question {i} was skipped: {question}")
    
    # Log collected feedback
    if feedback:
        logger.log_feedback(
            questions=[qa.question for qa in feedback],
            answers=[qa.answer for qa in feedback]
        )
        logger.log_info(f"Collected {len(feedback)} answers from {len(questions)} questions")
    else:
        logger.log_warning("No feedback answers were provided")
            
    return feedback

async def generate_serp_queries(query: str, num_queries: int = 3, learnings: List[str] = None, feedback: List[FeedbackQA] = None) -> List[SerpQuery]:
    """Generate SERP queries based on user input and feedback"""
    logger.log_info(f"Generating SERP queries for: {query}")
    learnings = learnings or []
    
    # Build context from feedback
    feedback_context = ""
    if feedback:
        feedback_context = "\n\nUser provided these clarifications:\n" + "\n".join(
            f"Q: {qa.question}\nA: {qa.answer}" for qa in feedback
        )
        logger.log_debug("Including feedback context in query generation")
        
    prompt = f"""Given the following prompt from the user, generate a list of SERP queries to research the topic. 
    Return a maximum of {num_queries} queries in a structured format. Each query should be on a new line starting with a number and include both the query and its research goal. Make sure each query is unique and not similar to others.
    Use the user's clarifications to guide the query generation and focus on their specific interests.
    
    <prompt>{query}</prompt>
    {feedback_context}
    
    {'Here are some learnings from previous research:' + ''.join(f'<learning>{learning}</learning>' for learning in learnings) if learnings else ''}"""
    
    response = await openai_client.generate_text(prompt, system_prompt())
    
    queries = []
    current_query = None
    
    for line in response.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        query_match = re.match(r'^\d+[\.\)]?\s*\*?\*?"?([^"]+)"?\*?\*?', line)
        if query_match:
            if current_query:
                queries.append(current_query)
            current_query = SerpQuery(
                query=query_match.group(1).strip(),
                research_goal=""
            )
        elif current_query and line.lower().startswith('focus:'):
            current_query.research_goal = re.sub(r'^\*|\*$', '', line.split('Focus:')[1].strip())
            queries.append(current_query)
            current_query = None
        elif current_query and not line.startswith(('*', '-')):
            current_query.research_goal = re.sub(r'^\*|\*$', '', line.strip())
            queries.append(current_query)
            current_query = None
    
    if current_query:
        queries.append(current_query)
        
    queries = queries[:num_queries]
    logger.log_info(f"Generated {len(queries)} SERP queries")
    for i, q in enumerate(queries, 1):
        logger.log_debug(f"Query {i}:\nQuery: {q.query}\nGoal: {q.research_goal}")
    
    return queries

async def process_serp_result(query: str, result: Dict, num_learnings: int = 3, num_follow_up: int = 3) -> Dict:
    """Process search results and extract learnings"""
    logger.log_info(f"Processing search results for query: {query}")
    contents = [
        trim_prompt(item.get('markdown', ''), 25000)
        for item in result.get('data', [])
        if item.get('markdown')
    ]
    
    logger.log_debug(f"Found {len(contents)} content items to process")

    prompt = f"""Given the following contents from a SERP search for the query <query>{query}</query>, 
    generate a list of learnings from the contents. Return a maximum of {num_learnings} learnings, but feel free to return less 
    if the contents are clear. Make sure each learning is unique and information dense. Include entities, metrics, 
    numbers, and dates when available. The learnings will be used to research the topic further.

    <contents>{''.join(f'<content>{content}</content>' for content in contents)}</contents>"""

    response = await openai_client.generate_text(prompt, system_prompt())
    
    learnings = []
    follow_up_questions = []
    
    current_section = None
    for line in response.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        if line.lower().startswith('learning'):
            current_section = 'learnings'
        elif line.lower().startswith('follow-up'):
            current_section = 'follow_up'
        elif current_section == 'learnings':
            learnings.append(line)
        elif current_section == 'follow_up':
            follow_up_questions.append(line)

    learnings = learnings[:num_learnings]
    follow_up_questions = follow_up_questions[:num_follow_up]
    
    logger.log_info(f"Extracted {len(learnings)} learnings and {len(follow_up_questions)} follow-up questions")
    logger.log_debug("Learnings:\n" + "\n".join(f"  - {l}" for l in learnings))
    logger.log_debug("Follow-up questions:\n" + "\n".join(f"  - {q}" for q in follow_up_questions))

    return {
        'learnings': learnings,
        'followUpQuestions': follow_up_questions
    }

async def write_final_report(prompt: str, learnings: List[str], visited_urls: List[str]) -> str:
    """Generate final research report"""
    logger.log_info("Starting final report generation")
    try:
        learnings_string = trim_prompt(
            ''.join(f'<learning>{learning}</learning>' for learning in learnings),
            150000
        )
        
        logger.log_debug(f"Using {len(learnings)} learnings in report generation")

        prompt = f"""Given the following prompt from the user, write a final report on the topic using the learnings from research. 
        Make it as detailed as possible, aim for 3 or more pages, include ALL the learnings from research. 
        Return the report in markdown format:

        <prompt>{prompt}</prompt>

        Here are all the learnings from previous research:

        <learnings>{learnings_string}</learnings>"""

        report = await openai_client.generate_text(prompt, system_prompt())
        urls_section = "\n\n## Sources\n\n" + "\n".join(f"- {url}" for url in visited_urls)
        final_report = report + urls_section
        
        logger.log_info("Final report generated successfully")
        logger.log_debug(f"Report length: {len(final_report)} characters")
        logger.log_debug(f"Number of sources: {len(visited_urls)}")
        
        return final_report
    except Exception as e:
        error_message = str(e)
        logger.log_error(e, "Error generating final report")
        error_report = f"Error generating report: {error_message}\n\n## Sources\n\n" + "\n".join(f"- {url}" for url in visited_urls)
        return error_report

async def deep_research(
    query: str,
    breadth: int,
    depth: int,
    learnings: List[str] = None,
    visited_urls: List[str] = None,
    feedback: List[FeedbackQA] = None,
    progress_callback: Optional[callable] = None,
    tracker: Optional[ResearchTracker] = None
) -> Dict:
    """
    Perform deep research on a topic with research limiting measures
    """
    logger.log_info(f"Starting deep research: depth={depth}, breadth={breadth}")
    logger.log_debug(f"Initial query: {query}")
    
    learnings = learnings or []
    visited_urls = visited_urls or []

    # Check research limits if tracker is provided
    if tracker:
        should_continue, stop_reason = tracker.should_continue()
        if not should_continue:
            logger.log_info(f"Research stopped: {stop_reason}")
            return {
                'learnings': list(tracker.learning_history),
                'visited_urls': visited_urls,
                'stop_reason': stop_reason
            }

        # Check if query is too similar to previous ones
        if not tracker.add_query(query):
            logger.log_info("Query too similar to previous queries, skipping")
            return {
                'learnings': list(tracker.learning_history),
                'visited_urls': visited_urls,
                'stop_reason': "Query too similar to previous queries"
            }
    
    progress = ResearchProgress(
        current_depth=depth,
        total_depth=depth,
        current_breadth=breadth,
        total_breadth=breadth,
        current_query=None,
        total_queries=0,
        completed_queries=0
    )

    serp_queries = await generate_serp_queries(query, breadth, learnings, feedback)
    
    progress.total_queries = len(serp_queries)
    progress.current_query = serp_queries[0].query if serp_queries else None
    
    if progress_callback:
        progress_callback(progress)

    async def process_query(serp_query: SerpQuery) -> Dict:
        try:
            async with semaphore:
                logger.log_info(f"Processing query: {serp_query.query}")
                logger.log_debug(f"Research goal: {serp_query.research_goal}")
                
                search_results = await firecrawl_client.search(
                    serp_query.query,
                    timeout=15000,
                    limit=5
                )
                
                new_urls = [item['url'] for item in search_results.get('data', [])]
                new_breadth = breadth // 2
                new_depth = depth - 1
                
                new_learnings = await process_serp_result(
                    serp_query.query,
                    search_results,
                    num_follow_up=new_breadth
                )
                
                progress.completed_queries += 1
                if progress_callback:
                    progress_callback(progress)

                all_learnings = learnings + new_learnings.get('learnings', [])
                all_urls = visited_urls + new_urls

                if new_depth > 0:
                    logger.log_info(
                        f"Continuing research: depth={new_depth}, breadth={new_breadth}"
                    )
                    
                    progress.current_depth = new_depth
                    progress.current_breadth = new_breadth
                    progress.current_query = serp_query.query
                    if progress_callback:
                        progress_callback(progress)
                    
                    next_query = f"""
                    Previous research goal: {serp_query.research_goal}
                    Follow-up research directions: {' '.join(new_learnings['followUpQuestions'])}
                    """.strip()
                    
                    # Check new information ratio before continuing research
                    if tracker:
                        new_info_ratio = tracker.add_learnings(new_learnings.get('learnings', []))
                        if new_info_ratio < tracker.limits.min_new_info_ratio:
                            logger.log_info(f"Insufficient new information ratio ({new_info_ratio:.2f}), stopping branch")
                            return {
                                'learnings': all_learnings,
                                'visited_urls': all_urls,
                                'stop_reason': "Insufficient new information"
                            }

                    return await deep_research(
                        query=next_query,
                        breadth=new_breadth,
                        depth=new_depth,
                        learnings=all_learnings,
                        visited_urls=all_urls,
                        feedback=feedback,
                        progress_callback=progress_callback,
                        tracker=tracker
                    )
                else:
                    logger.log_info("Reached maximum depth, returning results")
                    progress.current_depth = 0
                    progress.current_query = serp_query.query
                    if progress_callback:
                        progress_callback(progress)
                        
                    return {
                        'learnings': all_learnings,
                        'visited_urls': all_urls
                    }

        except Exception as e:
            if 'Timeout' in str(e):
                logger.log_error(e, f"Timeout error for query: {serp_query.query}")
            else:
                logger.log_error(e, f"Error processing query: {serp_query.query}")
            return {
                'learnings': [],
                'visited_urls': []
            }

    tasks = [process_query(serp_query) for serp_query in serp_queries]
    results = await asyncio.gather(*tasks)
    
    all_learnings = []
    all_urls = []
    for result in results:
        if result:
            all_learnings.extend(result.get('learnings', []))
            all_urls.extend(result.get('visited_urls', []))

    # Remove duplicates while preserving order
    all_learnings = list(dict.fromkeys(all_learnings))
    all_urls = list(dict.fromkeys(all_urls))
    
    logger.log_info(f"Research complete: {len(all_learnings)} learnings, {len(all_urls)} sources")
    
    return {
        'learnings': all_learnings,
        'visited_urls': all_urls
    }

async def main():
    logger.log_info("Starting Deep Research session")
    
    # Get user input
    query = input("What would you like to research? ")
    logger.log_info(f"Initial query: {query}")
    
    # Generate and collect feedback
    print("\nGenerating clarifying questions...\n")
    feedback_questions = await generate_feedback(query)
    feedback = await collect_feedback(feedback_questions)
    
    breadth = int(input("\nEnter research breadth (recommended 2-10, default 4): ") or "4")
    depth = int(input("Enter research depth (recommended 1-5, default 2): ") or "2")
    
    logger.log_info(f"Research parameters: depth={depth}, breadth={breadth}")
    print("\nResearching your topic...\n")

    # Initialize research limits and tracker
    limits = ResearchLimits()
    tracker = ResearchTracker(limits)
    
    # Perform deep research with feedback and tracker
    result = await deep_research(
        query=query,
        breadth=breadth,
        depth=depth,
        feedback=feedback,
        progress_callback=output_manager.update_progress,
        tracker=tracker
    )

    if isinstance(result, dict) and 'stop_reason' in result:
        print(f"\nResearch stopped: {result['stop_reason']}")

    logger.log_info("Research complete, generating final report")
    print("\n\nWriting final report...")
    
    # Generate and save final report
    report = await write_final_report(
        query,
        result['learnings'],
        result['visited_urls']
    )

    with open('report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.log_info("Report saved to report.md")
    print(f"\n\nFinal Report:\n\n{report}")
    print('\nReport has been saved to report.md')

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.log_info("Process interrupted by user")
    except Exception as e:
        logger.log_critical("Unexpected error in main process", e)
        raise