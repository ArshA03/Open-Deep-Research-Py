import requests
import json
import sys
import time

url = "https://openrouter.ai/api/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {"API_KEY"}",
    "Content-Type": "application/json"
}
payload = {
    "model": "deepseek/deepseek-r1:free",
    "messages": [
        {"role": "user", "content": "How would you build the world's tallest skyscraper?"}
    ],
    "include_reasoning": True,
    "stream": True,
    "provider": {
        "order": [
            'Targon',
            'Chutes',
            'Azure'
        ],
        "allow_fallbacks": False
    },
    "plugins": [{ "id": "web" }]
}

try:
    with requests.post(url, headers=headers, json=payload, stream=True) as response:
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            sys.exit(1)
            
        # Initialize variables for tracking state and buffering content
        from collections import deque
        
        in_reasoning = True
        content_buffer = deque()
        consecutive_empty_reasoning = 0
        MAX_EMPTY_REASONING = 10  # Number of empty reasoning chunks before switching to content
        
        for chunk in response.iter_lines():
            if chunk:
                decoded_chunk = chunk.decode('utf-8')
                if decoded_chunk.startswith("data: "):
                    try:
                        data = json.loads(decoded_chunk[6:])
                        if 'choices' in data and len(data['choices']) > 0:
                            content = data['choices'][0].get('delta', {}).get('content', '')
                            reasoning = data['choices'][0].get('delta', {}).get('reasoning', '')

                            # Handle reasoning phase
                            if in_reasoning:
                                if reasoning:
                                    print(reasoning, end='', flush=True)
                                    consecutive_empty_reasoning = 0
                                else:
                                    consecutive_empty_reasoning += 1
                                
                                # Check for transition conditions
                                if consecutive_empty_reasoning >= MAX_EMPTY_REASONING:
                                    in_reasoning = False
                                    print("\n\n---\n\n")  # Add newline after reasoning
                                    # Flush content buffer
                                    while content_buffer:
                                        buffered_content = content_buffer.popleft()
                                        if buffered_content.strip():
                                            print(buffered_content, end='', flush=True)
                            
                            # Buffer content while in reasoning phase, print directly otherwise
                            if content:
                                if in_reasoning:
                                    content_buffer.append(content)
                                else:
                                    if content.strip():
                                        print(content, end='', flush=True)
                                
                    except json.JSONDecodeError:
                        continue
except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
    sys.exit(1)