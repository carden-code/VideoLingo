# VideoLingo: Известные проблемы и улучшения

## Критические исправления (уже сделаны)

### 1. `load_key()` с дефолтными значениями
**Проблема:** Функция `load_key()` принимает только 1 аргумент, но в коде вызывалась с 2.

**Файлы:**
- `core/_4_2_translate.py:31` — `load_key("verify_translation", False)`
- `core/_4_2_translate.py:60` — `load_key("target_language", "English")`
- `core/tts_backend/cosyvoice3_tts.py:339` — `load_key("target_language", "English")`

**Решение:** Обёртка в try-except:
```python
try:
    value = load_key("key")
except KeyError:
    value = "default"
```

**Статус:** ✅ Исправлено в коммитах `cb3cd72`, `f13aac4`

---

### 2. DataFrame `.loc[]` для присвоения списков
**Проблема:** `tasks_df.loc[mask, 'column'] = [list_value]` вызывает ошибку pandas.

**Ошибка:**
```
ValueError: Must have equal len keys and value when setting with an ndarray
```

**Файл:** `core/_10_gen_audio.py:277, 300-303`

**Решение:** Использовать `.at[]` для присвоения в конкретную ячейку:
```python
# Было:
tasks_df.loc[tasks_df['number'] == number, 'lines'] = [out_lines]

# Стало:
idx = tasks_df.index[tasks_df['number'] == number][0]
tasks_df.at[idx, 'lines'] = out_lines
```

**Статус:** ✅ Исправлено в коммите `681b3e2`

---

## Проблемы с LLM ответами (требуют улучшения)

### 3. LLM возвращает массив вместо объекта
**Проблема:** Модель возвращает JSON `[...]` вместо `{...}`, код падает при вызове `.keys()` или `.get()`.

**Ошибки:**
```
'list' object has no attribute 'keys'
'list' object has no attribute 'get'
```

**Примеры промптов где возникает:**
- Перевод (`translate_lines.py`)
- Верификация перевода (`_4_2_translate.py:verify_translation_quality`)
- Выравнивание субтитров (`_5_split_sub.py:align_subs`)

**Текущее поведение:** retry механизм повторяет запрос, но не гарантирует правильный формат.

**Рекомендуемое решение — Structured Outputs:**
```python
# Вариант 1: JSON Schema на уровне API
response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "translation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "translations": {"type": "array", "items": {"type": "object"}}
            },
            "required": ["translations"]
        }
    }
}

# Вариант 2: Pydantic модели (beta)
from pydantic import BaseModel

class TranslationResponse(BaseModel):
    translations: list[dict]

response = client.beta.chat.completions.parse(
    model=model,
    messages=messages,
    response_format=TranslationResponse
)
```

**Ограничения:**
- Требует поддержки со стороны провайдера (OpenRouter/DeepSeek могут не поддерживать)
- `json_schema` работает только с OpenAI моделями

**Альтернатива — валидация на стороне клиента:**
```python
def validate_json_response(resp):
    if isinstance(resp, list):
        # Попытка извлечь объект из массива
        if len(resp) == 1 and isinstance(resp[0], dict):
            return resp[0]
        raise ValueError("Expected JSON object, got array")
    return resp
```

**Статус:** ⚠️ Требует реализации

---

### 4. Отсутствующие ключи в JSON ответе
**Проблема:** LLM не возвращает все необходимые ключи (например `target_part_2`).

**Ошибка:**
```
Error in split_align_subs: 'target_part_2', retry: 1/0
```

**Файл:** `core/_5_split_sub.py:align_subs`

**Причина:**
- Модель не следует инструкциям промпта
- retry=0 означает отсутствие повторных попыток

**Решение:**
1. Увеличить retry в декораторе
2. Добавить валидацию обязательных ключей
3. Использовать Structured Outputs (см. выше)

```python
REQUIRED_KEYS = ['source_part_1', 'source_part_2', 'target_part_1', 'target_part_2']

def validate_split_response(resp):
    missing = [k for k in REQUIRED_KEYS if k not in resp]
    if missing:
        raise ValueError(f"Missing keys: {missing}")
    return resp
```

**Статус:** ⚠️ Требует реализации

---

## Архитектурные улучшения

### 5. Централизованная валидация JSON ответов
**Текущее состояние:** Каждая функция сама парсит и валидирует JSON.

**Рекомендация:** Создать единый модуль для работы с LLM ответами:

```python
# core/utils/llm_response.py

from typing import Type, TypeVar
from pydantic import BaseModel, ValidationError

T = TypeVar('T', bound=BaseModel)

def parse_llm_response(
    response: str | dict | list,
    schema: Type[T] = None,
    required_keys: list[str] = None
) -> T | dict:
    """
    Парсит и валидирует ответ LLM.

    Args:
        response: Сырой ответ от LLM
        schema: Pydantic модель для валидации
        required_keys: Список обязательных ключей (если нет schema)

    Returns:
        Валидированный объект

    Raises:
        ValueError: Если валидация не прошла
    """
    # Преобразование списка в объект если нужно
    if isinstance(response, list):
        if len(response) == 1 and isinstance(response[0], dict):
            response = response[0]
        else:
            raise ValueError(f"Expected object, got list with {len(response)} items")

    # Валидация через Pydantic
    if schema:
        try:
            return schema.model_validate(response)
        except ValidationError as e:
            raise ValueError(f"Schema validation failed: {e}")

    # Валидация обязательных ключей
    if required_keys:
        missing = [k for k in required_keys if k not in response]
        if missing:
            raise ValueError(f"Missing required keys: {missing}")

    return response
```

**Статус:** 💡 Рекомендация

---

### 6. Добавить поддержку дефолтных значений в `load_key()`
**Текущее состояние:** `load_key()` выбрасывает `KeyError` при отсутствии ключа.

**Рекомендация:** Добавить опциональный параметр `default`:

```python
# core/utils/config_utils.py

def load_key(key, default=_UNSET):
    try:
        # ... existing code ...
        return value
    except KeyError:
        if default is not _UNSET:
            return default
        raise

_UNSET = object()  # Sentinel для различия default=None и отсутствия default
```

**Статус:** 💡 Рекомендация

---

## HTTP Proxy поддержка

### 7. Proxy для LLM запросов
**Реализовано:**
- Добавлена настройка `api.proxy` в `config.yaml`
- Используется `DefaultHttpxClient` с proxy параметром

**Формат:**
```yaml
api:
  proxy: 'http://username:password@host:port'
```

**Файл:** `core/utils/ask_gpt.py:61-70`

**Статус:** ✅ Реализовано в коммите `59f2631`

---

## Тестирование

### Рекомендуемые тесты для добавления:

1. **test_load_key_with_missing_key** — проверка KeyError
2. **test_json_response_validation** — валидация разных форматов JSON
3. **test_dataframe_list_assignment** — корректное присвоение списков в DataFrame
4. **test_proxy_configuration** — проверка инициализации с прокси

---

*Документ создан: 2024-12-23*
*Последнее обновление: после исправления DataFrame assignment*
