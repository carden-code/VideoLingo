import json


def parse_parent_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith('['):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
            return [str(parsed)]
        except json.JSONDecodeError:
            pass
    return [part.strip() for part in text.split(',') if part.strip()]


def encode_parent_list(values):
    if values is None:
        values = []
    if isinstance(values, tuple):
        values = list(values)
    return json.dumps(values, ensure_ascii=True)
