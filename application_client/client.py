import httpx

INTERFACE_URL = "http://localhost:8000"


def send_request(prompt: str, max_tokens: int = 256, temperature: float = 1.0) -> dict:
    with httpx.Client() as client:
        resp = client.post(
            f"{INTERFACE_URL}/generate",
            json={"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature},
            timeout=10.0,
        )
        resp.raise_for_status()
        return resp.json()


def main():
    samples = [
        ("Summarise this medical report for me.", 128, 0.7),
        ("Translate to French: Hello world", 64, 1.0),
        ("Write a Python function to reverse a string.", 256, 0.9),
    ]
    for prompt, max_tokens, temp in samples:
        result = send_request(prompt, max_tokens, temp)
        print(f"Queued: request_id={result['request_id']}  status={result['status']}")


if __name__ == "__main__":
    main()
