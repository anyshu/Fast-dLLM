import json
import time
import uuid
import threading
from typing import Any, Dict, Iterable, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from app import (
    device_accelerated,
    generate_response_with_visualization_fast_dllm,
    model_accelerated,
    tokenizer,
)


SYSTEM_FINGERPRINT = "fast-dllm-diffusion-v2-api"
STREAM_MEDIA_TYPE = "text/event-stream"
STOP_SIGNAL = "data: [DONE]\n\n"

app = FastAPI(
    title="Fast-dLLM v2 OpenAI-Compatible API",
    description="OpenAI chat completions compatible endpoint backed by Fast-dLLM diffusion model",
    version="1.0.0",
)

generation_lock = threading.Lock()


class ChatCompletionMessage(BaseModel):
    role: str = Field(..., description="Role of the message author")
    content: str = Field(..., description="Message content")


class ExtraBody(BaseModel):
    diffusion_block_length: Optional[int] = Field(default=32, description="Diffusion block size")
    diffusion_threshold: Optional[float] = Field(default=0.5, description="Diffusion sampling threshold")


class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="Model identifier")
    messages: List[ChatCompletionMessage] = Field(..., description="Conversation history")
    max_tokens: int = Field(default=1024, description="Maximum number of generated tokens")
    temperature: float = Field(default=0.0, description="Sampling temperature")
    top_p: float = Field(default=0.95, description="Top-p sampling parameter")
    stream: bool = Field(default=False, description="Enable Server-Sent Events streaming")
    extra_body: Optional[ExtraBody] = Field(default=None, description="Diffusion-specific overrides")


class ChatCompletionChoice(BaseModel):
    index: int
    message: Dict[str, Any]
    finish_reason: Optional[str]


class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    system_fingerprint: str
    choices: List[ChatCompletionChoice]
    usage: Dict[str, Any]


def _generate_completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:32]}"


def _normalize_state_entry(entry: Any) -> Any:
    if isinstance(entry, tuple):
        token_text, color = entry
        return token_text, color
    return entry, None


from itertools import zip_longest
from typing import List, Any, Dict

def _compute_state_diff(prev_state: List[Any], new_state: List[Any]) -> List[Dict[str, Any]]:
    # Early return if states are identical (before normalization)
    if prev_state == new_state:
        return []
    
    # Normalize entries
    prev_norm = [_normalize_state_entry(e) for e in prev_state]
    new_norm = [_normalize_state_entry(e) for e in new_state]
    
    # Early return if normalized states are identical
    if prev_norm == new_norm:
        return []
    
    changes: List[Dict[str, Any]] = []
    
    # Use zip_longest to handle different length lists
    for idx, (prev_entry, new_entry) in enumerate(zip_longest(prev_norm, new_norm, fillvalue=(None, None))):
        prev_token = prev_entry[0]
        new_token = new_entry[0]
        
        # Skip unchanged entries
        if prev_token == new_token:
            continue
        
        # Handle deletions (new token is None)
        if new_token is None:
            if prev_token and prev_token != "[MASK]":
                changes.append({"position": idx, "text": "", "replace_length": 1})
            continue
        
        # Skip invalid new tokens
        if not new_token or new_token == "[MASK]":
            continue
        
        # Handle insertions and replacements
        change = {"position": idx, "text": new_token}
        if prev_token is not None:
            change["replace_length"] = 1
        changes.append(change)
    
    return changes



def _format_messages(messages: List[ChatCompletionMessage]) -> List[Dict[str, str]]:
    return [{"role": msg.role, "content": msg.content} for msg in messages]


def _stream_fast_dllm(
    request: ChatCompletionRequest,
    messages: List[Dict[str, str]],
    completion_id: str,
    created: int,
    prompt_tokens: int,
) -> Iterable[str]:
    max_tokens = request.max_tokens
    temperature = request.temperature
    top_p = request.top_p

    block_length = request.extra_body.diffusion_block_length if request.extra_body else 32
    if block_length > 32:
        block_length = 32
    if block_length < 4:
        block_length = 4
    threshold = request.extra_body.diffusion_threshold if request.extra_body else 0.95

    previous_state: List[Any] = []
    step_index = 0
    diffusion_start = time.time()
    completion_tokens = 0
    first_item = True
    prefill_time = 0.0
    generate_time = 0.0

    try:
        generator = generate_response_with_visualization_fast_dllm(
            model_accelerated,
            tokenizer,
            device_accelerated,
            messages,
            max_new_tokens=max_tokens,
            temperature=temperature,
            block_length=block_length,
            threshold=threshold,
            top_p=top_p,
        )

        for item in generator:
            # print(f"item: {type(item)}")
            if isinstance(item, list):
                if first_item:
                    prefill_time = time.time() - diffusion_start
                    generate_start = time.time()
                    first_item = False
                changes = _compute_state_diff(previous_state, item)
                if not changes:
                    previous_state = list(item)
                    continue

                previous_state = list(item)
                step_index += 1
                # print(f"item, step_index: {step_index}, changes: {changes}")

                # current_text = "".join([e[0] if isinstance(e, tuple) and e[0] else str(e) if e else "" for e in previous_state if e and (isinstance(e, tuple) and e[0] != "[MASK]" or str(e) != "[MASK]")])
                # completion_tokens = len(tokenizer.encode(current_text, add_special_tokens=False))

                delta = {
                    "content": changes,
                    "step": step_index,
                    "max_step": max_tokens,
                }

                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "system_fingerprint": SYSTEM_FINGERPRINT,
                    "choices": [
                        {
                            "index": 0,
                            "delta": delta,
                            "finish_reason": None,
                        }
                    ],
                    # "usage": {
                    #     "prompt_tokens": prompt_tokens,
                    #     "completion_tokens": completion_tokens,
                    #     "total_tokens": prompt_tokens + completion_tokens,
                    # },
                }

                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

            else:
                generate_time = time.time() - generate_start
                completion_tokens = len(tokenizer.encode(item, add_special_tokens=False))
                generation_time = prefill_time + generate_time
                generation_time_str = f"{generation_time:.3f}s"
                throughput = completion_tokens / generation_time if generation_time > 0 else 0
                throughput_str = f"{throughput:.3f} tokens/s"
                total_tokens = prompt_tokens + completion_tokens
                print(
                    "[Fast-dLLM Summary] "
                    f"prompt_tokens={prompt_tokens}, "
                    f"completion_tokens={completion_tokens}, "
                    f"total_tokens={total_tokens}, "
                    f"prefill_time={prefill_time:.3f}s, "
                    f"generate_time={generate_time:.3f}s, "
                    f"generation_time={generation_time_str}, "
                    f"throughput={throughput_str}",
                    flush=True
                )
                break

        completion_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model,
            "system_fingerprint": SYSTEM_FINGERPRINT,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": [],
                        "step": step_index,
                        "max_step": max_tokens,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "prefill_time": float(f"{prefill_time:.3f}"),
                "generate_time": float(f"{generate_time:.3f}"),
                "time_cost": float(f"{generation_time:.3f}")
            },
        }

        yield f"data: {json.dumps(completion_chunk, ensure_ascii=False)}\n\n"
        yield STOP_SIGNAL

    except Exception as exc:  # pragma: no cover - defensive
        error_payload = {
            "error": {
                "message": str(exc),
                "type": "internal_error",
            }
        }
        yield f"data: {json.dumps(error_payload, ensure_ascii=False)}\n\n"
        yield STOP_SIGNAL


def _run_fast_dllm(
    request: ChatCompletionRequest,
    messages: List[Dict[str, str]],
) -> tuple[str, float]:
    max_tokens = request.max_tokens
    temperature = request.temperature
    top_p = request.top_p

    block_length = request.extra_body.diffusion_block_length if request.extra_body else 32
    threshold = request.extra_body.diffusion_threshold if request.extra_body else 0.95

    start_time = time.time()
    first_item = True
    prefill_time = 0.0
    generate_time = 0.0
    generator = generate_response_with_visualization_fast_dllm(
        model_accelerated,
        tokenizer,
        device_accelerated,
        messages,
        max_new_tokens=max_tokens,
        temperature=temperature,
        block_length=block_length,
        threshold=threshold,
        top_p=top_p,
    )

    final_text = ""
    for item in generator:
        if isinstance(item, list):
            if first_item:
                prefill_time = time.time() - start_time
                generate_start = time.time()
                first_item = False
            continue
        final_text = item
        generate_time = time.time() - generate_start
        break

    generation_time = prefill_time + generate_time
    return final_text, generation_time, prefill_time, generate_time


@app.get("/health")
def health_check() -> Dict[str, Any]:  # pragma: no cover - simple health route
    return {
        "status": "ok",
        "device": device_accelerated,
        "model_loaded": model_accelerated is not None,
    }


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")
    print(request)
    formatted_messages = _format_messages(request.messages)
    prompt_token_ids = tokenizer.apply_chat_template(
        formatted_messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    prompt_tokens = len(prompt_token_ids)
    completion_id = _generate_completion_id()
    created = int(time.time())

    if request.stream:
        def event_stream() -> Iterable[str]:
            with generation_lock:
                yield from _stream_fast_dllm(
                    request,
                    formatted_messages,
                    completion_id,
                    created,
                    prompt_tokens,
                )

        return StreamingResponse(event_stream(), media_type=STREAM_MEDIA_TYPE)

    with generation_lock:
        final_text, generation_time, prefill_time, generate_time = _run_fast_dllm(request, formatted_messages)

    completion_tokens = len(tokenizer.encode(final_text, add_special_tokens=False))
    generation_time_str = f"{generation_time:.2f}s"
    throughput = completion_tokens / generation_time if generation_time > 0 else 0
    throughput_str = f"{throughput:.2f} tokens/s"
    total_tokens = prompt_tokens + completion_tokens

    print(
        "[Fast-dLLM Summary] "
        f"prompt_tokens={prompt_tokens}, "
        f"completion_tokens={completion_tokens}, "
        f"total_tokens={total_tokens}, "
        f"prefill_time={prefill_time:.2f}s, "
        f"generate_time={generate_time:.2f}s, "
        f"generation_time={generation_time_str}, "
        f"throughput={throughput_str}"
    )

    response = ChatCompletionResponse(
        id=completion_id,
        object="chat.completion",
        created=created,
        model=request.model,
        system_fingerprint=SYSTEM_FINGERPRINT,
        choices=[
            ChatCompletionChoice(
                index=0,
                message={"role": "assistant", "content": final_text},
                finish_reason="stop",
            )
        ],
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "prefill_time": float(f"{prefill_time:.3f}"),
            "generate_time": float(f"{generate_time:.3f}"),
            "time_cost": float(f"{generation_time:.3f}"),
        },
    )

    return JSONResponse(content=response.model_dump())


if __name__ == "__main__":  # pragma: no cover - manual launch helper
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000)