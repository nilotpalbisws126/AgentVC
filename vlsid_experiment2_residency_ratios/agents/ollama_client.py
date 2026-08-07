"""
Ollama API Client Wrapper for Remote GPU LLM Execution.
Connects to local or remote Ollama instance running Llama 3.1 8B/7B.
Supports prompt querying, response parsing, and execution logging.
Includes instant offline fallback detection to avoid TCP socket timeouts during dry-runs.
"""

import os
import json
import time
import urllib.request
import urllib.error
from typing import Dict, Any, Optional

class OllamaLLMClient:
    """Client interface for interacting with Ollama model on remote GPU."""

    def __init__(
        self,
        model_name: str = "llama3.1",
        host_url: str = "http://localhost:11434",
        temperature: float = 0.1,
        timeout: int = 60,
        logger: Optional[Any] = None
    ):
        self.model_name = os.getenv("OLLAMA_MODEL", model_name)
        self.host_url = os.getenv("OLLAMA_HOST", host_url).rstrip("/")
        self.temperature = temperature
        self.timeout = timeout
        self.logger = logger
        self.api_endpoint = f"{self.host_url}/api/generate"
        self.is_offline = False

    def _log(self, msg: str):
        if self.logger:
            self.logger.info(msg)
        else:
            print(f"[OllamaClient] {msg}")

    def query(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Sends a generation request to the Ollama server with automatic retries."""
        if self.is_offline:
            raise ConnectionError("Ollama server is offline. Instant fallback enabled.")

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "top_p": 0.9,
                "num_ctx": 4096
            }
        }
        if system_prompt:
            payload["system"] = system_prompt

        json_data = json.dumps(payload).encode("utf-8")
        
        max_attempts = 3
        last_exception = None

        for attempt in range(1, max_attempts + 1):
            req = urllib.request.Request(
                self.api_endpoint,
                data=json_data,
                headers={"Content-Type": "application/json"}
            )
            start_t = time.time()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as response:
                    resp_bytes = response.read()
                    resp_json = json.loads(resp_bytes.decode("utf-8"))
                    output_text = resp_json.get("response", "").strip()
                    duration = time.time() - start_t
                    self._log(f"Query completed in {duration:.2f}s using model '{self.model_name}' (Attempt {attempt})")
                    return output_text
            except (urllib.error.URLError, Exception) as e:
                last_exception = e
                self._log(f"Attempt {attempt}/{max_attempts} failed connecting to Ollama: {e}")
                if attempt < max_attempts:
                    time.sleep(2.0)  # Wait 2 seconds before retry

        self.is_offline = True
        err_msg = f"All {max_attempts} attempts failed connecting to Ollama at {self.api_endpoint}: {last_exception}."
        self._log(err_msg)
        raise ConnectionError(err_msg)

    def check_health(self) -> bool:
        """Verifies if Ollama server is reachable and model is loaded."""
        tags_url = f"{self.host_url}/api/tags"
        try:
            req = urllib.request.Request(tags_url)
            with urllib.request.urlopen(req, timeout=2) as response:
                data = json.loads(response.read().decode("utf-8"))
                models = [m.get("name") for m in data.get("models", [])]
                self._log(f"Ollama server active at {self.host_url}. Models available: {models}")
                self.is_offline = False
                return True
        except Exception as e:
            self.is_offline = True
            self._log(f"Ollama server health check failed at {self.host_url}: {e}. Instant offline mode active.")
            return False
