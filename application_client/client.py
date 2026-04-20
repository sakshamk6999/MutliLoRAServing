import time
import httpx

INTERFACE_URL = "http://localhost:8000"
POLL_INTERVAL = 0.5   # seconds between result polls
POLL_TIMEOUT  = 120.0 # give up after 2 minutes


def send_request(prompt: str, max_tokens: int = 256, temperature: float = 1.0) -> dict:
    """Queue a generation request and return immediately with request_id."""
    with httpx.Client() as client:
        resp = client.post(
            f"{INTERFACE_URL}/generate",
            json={"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature},
            timeout=10.0,
        )
        resp.raise_for_status()
        return resp.json()


def poll_result(request_id: str,
                poll_interval: float = POLL_INTERVAL,
                timeout: float = POLL_TIMEOUT) -> dict:
    """Poll GET /result/{request_id} until the result is ready or timeout expires."""
    deadline = time.monotonic() + timeout
    with httpx.Client() as client:
        while time.monotonic() < deadline:
            resp = client.get(
                f"{INTERFACE_URL}/result/{request_id}",
                timeout=10.0,
            )
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code != 202:
                resp.raise_for_status()
            time.sleep(poll_interval)
    raise TimeoutError(f"Result for {request_id} not ready after {timeout}s")


def generate_and_wait(prompt: str,
                      max_tokens: int = 256,
                      temperature: float = 1.0) -> dict:
    """Send request and block until the generated text is returned."""
    queued = send_request(prompt, max_tokens, temperature)
    print(f"Queued: request_id={queued['request_id']}")
    result = poll_result(queued["request_id"])
    return result


def main():
    samples = [
        ("Summarise this medical report for me.", 128, 0.7),
        ("Write a Python function to reverse a string.", 256, 0.9),
    ]
    for prompt, max_tokens, temp in samples:
        result = generate_and_wait(prompt, max_tokens, temp)
        print(f"  text:   {result['generated_text'][:120]}")
        print(f"  finish: {result['finish_reason']}")
        print()


if __name__ == "__main__":
    main()
