import os
import json
from threading import Lock
import json_repair
import httpx
from openai import OpenAI, DefaultHttpxClient
from core.utils.config_utils import load_key
from rich import print as rprint
from core.utils.decorator import except_handler

# ------------
# cache gpt response
# ------------

LOCK = Lock()
GPT_LOG_FOLDER = 'output/gpt_log'

def _save_cache(model, prompt, resp_content, resp_type, resp, message=None, log_title="default", json_schema=None):
    with LOCK:
        logs = []
        file = os.path.join(GPT_LOG_FOLDER, f"{log_title}.json")
        os.makedirs(os.path.dirname(file), exist_ok=True)
        if os.path.exists(file):
            with open(file, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        logs.append({
            "model": model,
            "prompt": prompt,
            "resp_content": resp_content,
            "resp_type": resp_type,
            "resp": resp,
            "message": message,
            "json_schema": json_schema
        })
        with open(file, 'w', encoding='utf-8') as f:
            json.dump(logs, f, ensure_ascii=False, indent=4)

def _load_cache(prompt, resp_type, log_title):
    with LOCK:
        file = os.path.join(GPT_LOG_FOLDER, f"{log_title}.json")
        if os.path.exists(file):
            with open(file, 'r', encoding='utf-8') as f:
                for item in json.load(f):
                    if item["prompt"] == prompt and item["resp_type"] == resp_type:
                        return item["resp"]
        return False

# ------------
# ask gpt once
# ------------

def _normalize_json_response(resp):
    if isinstance(resp, list):
        if len(resp) == 1 and isinstance(resp[0], dict):
            return resp[0]
        raise ValueError(f"Expected JSON object, got list with {len(resp)} items")
    return resp


@except_handler("GPT request failed", retry=5)
def ask_gpt(prompt, resp_type=None, valid_def=None, log_title="default", json_schema=None):
    if not load_key("api.key"):
        raise ValueError("API key is not set")
    # check cache
    cached = _load_cache(prompt, resp_type, log_title)
    if cached:
        rprint("use cache response")
        return cached

    model = load_key("api.model")
    base_url = load_key("api.base_url")
    if 'ark' in base_url:
        base_url = "https://ark.cn-beijing.volces.com/api/v3" # huoshan base url
    elif 'v1' not in base_url:
        base_url = base_url.strip('/') + '/v1'

    # Configure HTTP proxy if set
    try:
        proxy_url = load_key("api.proxy") or None
    except KeyError:
        proxy_url = None
    if proxy_url:
        http_client = DefaultHttpxClient(proxy=proxy_url)
        client = OpenAI(api_key=load_key("api.key"), base_url=base_url, http_client=http_client)
    else:
        client = OpenAI(api_key=load_key("api.key"), base_url=base_url)
    response_format = None
    if resp_type == "json" and load_key("api.llm_support_json"):
        try:
            structured = load_key("api.structured_output")
        except Exception:
            structured = "json"
        if structured == "schema" and json_schema:
            response_format = {
                "type": "json_schema",
                "json_schema": json_schema
            }
        elif structured == "json":
            response_format = {"type": "json_object"}

    messages = [{"role": "user", "content": prompt}]

    params = dict(
        model=model,
        messages=messages,
        response_format=response_format,
        timeout=300
    )
    try:
        resp_raw = client.chat.completions.create(**params)
    except Exception as exc:
        if response_format and response_format.get("type") == "json_schema":
            params["response_format"] = {"type": "json_object"}
            resp_raw = client.chat.completions.create(**params)
        else:
            raise exc

    # process and return full result
    resp_content = resp_raw.choices[0].message.content
    if resp_type == "json":
        resp = json_repair.loads(resp_content)
        resp = _normalize_json_response(resp)
    else:
        resp = resp_content
    
    # check if the response format is valid
    if valid_def:
        valid_resp = valid_def(resp)
        if valid_resp['status'] != 'success':
            _save_cache(model, prompt, resp_content, resp_type, resp, log_title="error", message=valid_resp['message'], json_schema=json_schema)
            raise ValueError(f"❎ API response error: {valid_resp['message']}")

    _save_cache(model, prompt, resp_content, resp_type, resp, log_title=log_title, json_schema=json_schema)
    return resp


if __name__ == '__main__':
    from rich import print as rprint
    
    result = ask_gpt("""test respond ```json\n{\"code\": 200, \"message\": \"success\"}\n```""", resp_type="json")
    rprint(f"Test json output result: {result}")
