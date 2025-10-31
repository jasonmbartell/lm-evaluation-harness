# Gemini API - OpenAI Compatibility Guide

This document summarizes how to use Google's Gemini API with OpenAI-compatible clients and libraries.

## Overview

Google Gemini models are accessible through OpenAI libraries (Python and TypeScript/JavaScript) and REST API by modifying only three configuration parameters. This allows you to use existing OpenAI-compatible tools with Gemini models.

**Current Status**: Beta - Google is actively extending feature support for OpenAI compatibility.

## Quick Setup

To switch from OpenAI to Gemini, you need to change three things:

1. **API Key**: Replace your OpenAI API key with a Gemini API key
2. **Base URL**: Point to Google's endpoint
3. **Model Name**: Use a Gemini model identifier

### Getting a Gemini API Key

Obtain your API key from: https://aistudio.google.com/apikey

## Configuration

### Python (OpenAI Python Library)

```python
from openai import OpenAI

client = OpenAI(
    api_key="YOUR_GEMINI_API_KEY",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ]
)
```

### JavaScript/TypeScript (OpenAI Node Library)

```javascript
import OpenAI from "openai";

const openai = new OpenAI({
    apiKey: "YOUR_GEMINI_API_KEY",
    baseURL: "https://generativelanguage.googleapis.com/v1beta/openai/"
});

const completion = await openai.chat.completions.create({
    model: "gemini-2.5-flash",
    messages: [
        { role: "user", content: "Hello, how are you?" }
    ]
});
```

### REST API

```bash
curl https://generativelanguage.googleapis.com/v1beta/openai/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_GEMINI_API_KEY" \
  -d '{
    "model": "gemini-2.5-flash",
    "messages": [
      {
        "role": "user",
        "content": "Hello, how are you?"
      }
    ]
  }'
```

## Available Models

### Text Generation Models

| Model | Description |
|-------|-------------|
| `gemini-2.5-flash` | Fast model with reasoning capabilities |
| `gemini-2.0-flash` | Standard fast model |
| `gemini-2.5-pro` | Advanced model for complex tasks |

### Specialized Models

| Model | Use Case |
|-------|----------|
| `imagen-3.0-generate-002` | Image generation (paid tier only) |
| `gemini-embedding-001` | Text embeddings |

## Supported Features

### Core Capabilities

✅ **Text Generation**
- Standard completions
- Streaming responses
- Multi-turn conversations

✅ **Vision Tasks**
- Image understanding
- Image analysis
- Image generation (Imagen models, paid tier)

✅ **Audio Processing**
- Audio analysis
- Transcription

✅ **Advanced Features**
- Function calling with structured parameters
- Structured JSON output (with schema validation)
- Text embeddings
- Batch API operations

### Gemini-Specific Features

#### Extended Reasoning

Gemini 2.5 models support extended reasoning with the `reasoning_effort` parameter:

```python
response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[{"role": "user", "content": "Solve this complex problem..."}],
    extra_body={
        "reasoning_effort": "high"  # Options: "low", "medium", "high", "none"
    }
)
```

#### Advanced Configuration via `extra_body`

The `extra_body` parameter enables Gemini-specific features not available in standard OpenAI API:
- Cached content
- Exact thinking budgets
- Other Gemini-specific parameters

```python
response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[...],
    extra_body={
        # Gemini-specific parameters here
    }
)
```

## Common Use Cases

### Streaming Responses

```python
stream = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### Function Calling

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[{"role": "user", "content": "What's the weather in San Francisco?"}],
    tools=tools
)
```

### Structured JSON Output

```python
from pydantic import BaseModel

class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]

completion = client.beta.chat.completions.parse(
    model="gemini-2.5-flash",
    messages=[
        {"role": "user", "content": "Schedule a team meeting next Tuesday with Alice and Bob"}
    ],
    response_format=CalendarEvent
)

event = completion.choices[0].message.parsed
```

### Vision (Image Analysis)

```python
response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/image.jpg"}
                }
            ]
        }
    ]
)
```

### Text Embeddings

```python
response = client.embeddings.create(
    model="gemini-embedding-001",
    input="Your text here"
)

embedding = response.data[0].embedding
```

### Batch API

```python
# Create batch file with requests
batch_input = {
    "custom_id": "request-1",
    "method": "POST",
    "url": "/v1/chat/completions",
    "body": {
        "model": "gemini-2.5-flash",
        "messages": [{"role": "user", "content": "Hello"}]
    }
}

# Submit batch
batch = client.batches.create(
    input_file_id="file-abc123",
    endpoint="/v1/chat/completions",
    completion_window="24h"
)

# Check status
batch_status = client.batches.retrieve(batch.id)
```

## Important Differences from OpenAI

### 1. Base URL
- **OpenAI**: `https://api.openai.com/v1/`
- **Gemini**: `https://generativelanguage.googleapis.com/v1beta/openai/`

### 2. API Key Format
- Gemini API keys are obtained from Google AI Studio
- No organization ID required

### 3. Model Names
- Use Gemini model identifiers (e.g., `gemini-2.5-flash` instead of `gpt-4`)

### 4. Extended Features
- `extra_body` parameter for Gemini-specific features
- `reasoning_effort` parameter for extended reasoning
- Different embedding model (`gemini-embedding-001` vs `text-embedding-ada-002`)

### 5. Image Generation
- Available only on paid tier
- Uses Imagen models instead of DALL-E

### 6. Beta Status
- Feature set is still expanding
- Some OpenAI features may not yet be supported
- Check documentation for latest feature availability

## Limitations & Considerations

1. **Beta Status**: The OpenAI compatibility layer is still in beta. Some features may be incomplete or change over time.

2. **Image Generation**: Requires a paid API tier (not available on free tier).

3. **Feature Parity**: Not all OpenAI API features may be available. Check the latest documentation for supported features.

4. **Pricing**: Different pricing structure than OpenAI. Check Google's pricing page for current rates.

5. **Rate Limits**: May differ from OpenAI's rate limits. Consult Google's documentation for specifics.

## Using with lm-evaluation-harness

To use Gemini models with the lm-evaluation-harness, use the custom `gemini-chat` model type which handles Gemini-specific API compatibility:

```bash
# Using the gemini-chat model type (recommended)
lm_eval --model gemini-chat \
    --model_args model=gemini-2.5-flash,base_url=https://generativelanguage.googleapis.com/v1beta/openai/chat/completions,tokenizer_backend=huggingface,tokenizer=Xenova/gpt-4 \
    --tasks hellaswag \
    --batch_size 1

# Set your API key as environment variable
export OPENAI_API_KEY=YOUR_GEMINI_API_KEY
```

The `gemini-chat` model type is a custom adapter that:
- Removes unsupported parameters (like `seed`) from API requests
- Is specifically designed for Gemini's OpenAI compatibility layer
- Located in `lm_eval/models/gemini_chat.py`

**Important Notes:**

*Base URL Configuration:*
- For `local-chat-completions`: Use the full endpoint URL including `/chat/completions`
- For `local-completions`: Use the full endpoint URL including `/completions`
- The base_url should be the complete endpoint path, not just the base domain

*Tokenizer Backend:*
- Use `tokenizer_backend=huggingface` with `tokenizer=Xenova/gpt-4` for Gemini models (recommended)
- Do NOT use `tokenizer_backend=tiktoken` - tiktoken doesn't recognize Gemini model names and will throw a KeyError
- Do NOT use `tokenizer_backend=remote` - Gemini's OpenAI compatibility layer doesn't support the `/tokenizer_info` endpoint
- The tokenizer is only used for local token counting and doesn't affect the actual API requests

Or for completions endpoint:

```bash
lm_eval --model local-completions \
    --model_args model=gemini-2.5-flash,base_url=https://generativelanguage.googleapis.com/v1beta/openai/completions,tokenizer_backend=huggingface,tokenizer=Xenova/gpt-4 \
    --tasks lambada_openai \
    --batch_size 1
```

### Python API Usage

```python
from lm_eval import simple_evaluate

results = simple_evaluate(
    model="gemini-chat",  # Use gemini-chat model type
    model_args={
        "model": "gemini-2.5-flash",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "tokenizer_backend": "huggingface",
        "tokenizer": "Xenova/gpt-4",
    },
    tasks=["mmlu_pro_physics"],
    num_fewshot=0,
    batch_size=1,
    apply_chat_template=True,
)
```

### Custom Model Implementation

The `gemini-chat` model type is implemented in `lm_eval/models/gemini_chat.py`. If you encounter additional unsupported parameters, you can modify this file to remove them:

```python
# In lm_eval/models/gemini_chat.py
def _create_payload(self, messages, generate=False, gen_kwargs=None, seed=1234, eos=None, **kwargs):
    payload = super()._create_payload(messages, generate, gen_kwargs, seed, eos, **kwargs)

    # Remove unsupported parameters for Gemini
    payload.pop("seed", None)
    payload.pop("other_unsupported_param", None)  # Add more as needed

    return payload
```

## Additional Resources

- **Get API Key**: https://aistudio.google.com/apikey
- **Official Documentation**: https://ai.google.dev/gemini-api/docs/openai
- **API Reference**: https://ai.google.dev/gemini-api/docs
- **Custom Model Code**: `lm_eval/models/gemini_chat.py`

## Troubleshooting

### Common Issues

**Authentication Errors**
- Verify your API key is correct
- Ensure the key is from Google AI Studio, not an OpenAI key
- Check that the base URL is correctly set

**Model Not Found**
- Use Gemini model names (e.g., `gemini-2.5-flash`), not OpenAI model names
- Verify the model is available in your region/tier

**Feature Not Supported**
- Check if the feature is listed as supported in the latest documentation
- Remember this is a beta feature - some capabilities may be coming soon

**Rate Limiting**
- Check your quota in Google AI Studio
- Consider upgrading to a paid tier if using free tier

**Tokenizer Errors**

*Error: `KeyError: 'Could not automatically map gemini-2.5-flash to a tokeniser'`*
- This occurs when using `tokenizer_backend=tiktoken`
- Solution: Use `tokenizer_backend=huggingface` with `tokenizer=Xenova/gpt-4` instead

*Error: `404 Client Error: Not Found for url: .../tokenizer_info`*
- This occurs when using `tokenizer_backend=remote`
- Gemini's OpenAI compatibility layer doesn't support the `/tokenizer_info` endpoint
- Solution: Use `tokenizer_backend=huggingface` with `tokenizer=Xenova/gpt-4` instead

*Error: `404 Client Error: Not Found for url: https://generativelanguage.googleapis.com/v1beta/openai/`*
- This occurs when the base_url doesn't include the full endpoint path
- The `local-chat-completions` model expects the full endpoint URL, not just the base path
- Solution: Use the full endpoint URL including `/chat/completions`:
  - Correct: `base_url=https://generativelanguage.googleapis.com/v1beta/openai/chat/completions`
  - Incorrect: `base_url=https://generativelanguage.googleapis.com/v1beta/openai/`

*Error: `Invalid JSON payload received. Unknown name "seed": Cannot find field.`*
- This occurs when using `local-chat-completions` model type with Gemini
- Gemini's OpenAI compatibility layer doesn't support the `seed` parameter
- Solution: Use the `gemini-chat` model type instead, which automatically removes unsupported parameters
