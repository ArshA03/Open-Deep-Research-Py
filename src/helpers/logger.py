import os
import sys
import json
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from typing import Any, Dict, Optional
import traceback

class LogManager:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Create log file with timestamp
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(log_dir, f"deepr_{self.timestamp}.log")
        
        # Set up logging
        self.logger = logging.getLogger('DeepResearch')
        self.logger.setLevel(logging.DEBUG)
        
        # Create formatters
        file_formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_formatter = logging.Formatter(
            '[%(levelname)s] %(message)s'
        )
        
        # File handler (with rotation)
        file_handler = RotatingFileHandler(
            self.log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Error log file
        error_handler = RotatingFileHandler(
            os.path.join(log_dir, f"deepr_errors_{self.timestamp}.log"),
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        self.logger.addHandler(error_handler)
        
        self.log_startup()
    
    def log_startup(self):
        """Log startup information"""
        self.logger.info("="*50)
        self.logger.info("Deep Research Session Started")
        self.logger.info(f"Timestamp: {self.timestamp}")
        self.logger.info(f"Log file: {self.log_file}")
        self.logger.info("="*50)
    
    def format_api_call(self, endpoint: str, method: str, params: Dict, response: Any, duration: float) -> str:
        """Format API call details for logging"""
        return f"""
API Call:
  Endpoint: {endpoint}
  Method: {method}
  Parameters: {json.dumps(params, indent=2)}
Response:
  Duration: {duration:.2f}s
  Content: {json.dumps(response, indent=2) if isinstance(response, (dict, list)) else str(response)}
"""

    def format_prompt(self, system: str, user: str, response: str, model: str, tokens: Optional[int] = None) -> str:
        """Format prompt details for logging"""
        return f"""
Prompt:
  Model: {model}
  System: {system}
  User: {user}
Response:
  Content: {response}
  Tokens: {tokens if tokens is not None else 'N/A'}
"""

    def log_api_call(self, endpoint: str, method: str, params: Dict, response: Any, duration: float):
        """Log API call details"""
        self.logger.debug(self.format_api_call(endpoint, method, params, response, duration))
    
    def log_prompt(self, system: str, user: str, response: str, model: str, tokens: Optional[int] = None):
        """Log prompt and response details"""
        self.logger.debug(self.format_prompt(system, user, response, model, tokens))
    
    def log_research_progress(self, depth: int, breadth: int, query: str, completed: int, total: int):
        """Log research progress"""
        self.logger.info(f"""
Research Progress:
  Depth: {depth}
  Breadth: {breadth}
  Current Query: {query}
  Progress: {completed}/{total} queries completed
""")
    
    def log_error(self, error: Exception, context: str = ""):
        """Log error with full stack trace and context"""
        error_msg = f"""
Error occurred: {context}
Type: {type(error).__name__}
Message: {str(error)}
Stack trace:
{traceback.format_exc()}
"""
        self.logger.error(error_msg)
    
    def log_warning(self, message: str, context: Dict = None):
        """Log warning with optional context"""
        if context:
            message = f"{message}\nContext: {json.dumps(context, indent=2)}"
        self.logger.warning(message)
    
    def log_info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def log_debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)
    
    def log_critical(self, message: str, error: Optional[Exception] = None):
        """Log critical error"""
        if error:
            message = f"{message}\nError: {type(error).__name__}: {str(error)}"
        self.logger.critical(message)
    
    def log_search_result(self, query: str, num_results: int, urls: list[str]):
        """Log search results"""
        self.logger.debug(f"""
Search Results:
  Query: {query}
  Number of results: {num_results}
  URLs found:
{chr(10).join('    - ' + url for url in urls)}
""")
    
    def log_learning(self, query: str, learnings: list[str]):
        """Log extracted learnings"""
        self.logger.debug(f"""
Learnings Extracted:
  Query: {query}
  Number of learnings: {len(learnings)}
  Content:
{chr(10).join('    - ' + learning for learning in learnings)}
""")
    
    def log_feedback(self, questions: list[str], answers: list[str]):
        """Log feedback Q&A"""
        self.logger.info(f"""
Feedback Collected:
{chr(10).join(f'  Q{i+1}: {q}\n  A{i+1}: {a}' for i, (q, a) in enumerate(zip(questions, answers)))}
""")
