#!/usr/bin/env python3
"""
OpenAI-compatible API for LLaDA Diffusion LLM
Supports streaming and non-streaming chat completions with diffusion model-specific parameters.
"""

import json
import time
import uuid
import asyncio
import traceback
from typing import List, Dict, Any, Optional, AsyncGenerator, Generator
from dataclasses import dataclass
from datetime import datetime

import torch
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

# Import the model and generation functions from app.py
from model.modeling_llada import LLaDAModelLM
from app import (
    generate_response_with_visualization_cache_and_parallel,
    add_gumbel_noise,
    get_num_transfer_tokens,
    get_transfer_index,
    MASK_TOKEN,
    MASK_ID
)

# Initialize device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load model and tokenizer (global variables for API)
tokenizer = None
model = None

def load_model():
    """Load the LLaDA model and tokenizer"""
    global tokenizer, model
    if tokenizer is None or model is None:
        print("Loading LLaDA model...")
        start_time = time.time()
        
        # Load tokenizer
        tokenizer_start = time.time()
        tokenizer = AutoTokenizer.from_pretrained('/mnt/data/models/GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)
        tokenizer_time = time.time() - tokenizer_start
        print(f"Tokenizer loaded in {tokenizer_time:.2f}s")
        
        # Load model
        model_start = time.time()
        model = LLaDAModelLM.from_pretrained('/mnt/data/models/GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, 
                                          torch_dtype=torch.bfloat16).to(device)
        model_time = time.time() - model_start
        print(f"Model loaded in {model_time:.2f}s")
        
        total_time = time.time() - start_time
        print(f"Total model loading time: {total_time:.2f}s")
        
        # Warm up the model with a dummy forward pass
        print("Warming up model...")
        warmup_start = time.time()
        with torch.no_grad():
            dummy_input = torch.randint(0, 1000, (1, 32)).to(device)
            _ = model(dummy_input)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
        warmup_time = time.time() - warmup_start
        print(f"Model warmup completed in {warmup_time:.2f}s")
        
        print("Model loaded and warmed up successfully!")

# Pydantic models for API
class Message(BaseModel):
    role: str = Field(..., description="Role of the message sender")
    content: str = Field(..., description="Content of the message")

class ExtraBody(BaseModel):
    diffusion_steps: int = Field(default=128, description="Number of diffusion denoising steps")
    diffusion_remasking: str = Field(default="low_confidence", description="Diffusion remasking algorithm")
    diffusion_block_length: int = Field(default=32, description="Block length for semi-autoregressive generation")
    diffusion_threshold: float = Field(default=0.9, description="Confidence threshold for token selection")

class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="Model to use for completion")
    messages: List[Message] = Field(..., description="List of messages in the conversation")
    max_tokens: int = Field(default=512, description="Maximum number of tokens to generate")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    stream: bool = Field(default=False, description="Whether to stream the response")
    extra_body: Optional[ExtraBody] = Field(default=None, description="Extra parameters for diffusion model")

class DiffusionContent(BaseModel):
    position: int = Field(..., description="Position of the generated token")
    text: str = Field(..., description="Generated text at this position")

class DiffusionDelta(BaseModel):
    content: List[DiffusionContent] = Field(default_factory=list, description="Generated content with positions")
    step: int = Field(..., description="Current diffusion step")
    max_step: int = Field(..., description="Maximum number of diffusion steps")

class Choice(BaseModel):
    index: int = Field(default=0, description="Choice index")
    delta: Optional[DiffusionDelta] = Field(default=None, description="Delta for streaming")
    message: Optional[Dict[str, Any]] = Field(default=None, description="Complete message for non-streaming")
    finish_reason: Optional[str] = Field(default=None, description="Reason for finishing")

class ChatCompletionResponse(BaseModel):
    id: str = Field(..., description="Unique identifier for the completion")
    object: str = Field(default="chat.completion", description="Object type")
    created: int = Field(..., description="Unix timestamp of creation")
    model: str = Field(..., description="Model used for completion")
    system_fingerprint: str = Field(default="llada-diffusion-api", description="System fingerprint")
    choices: List[Choice] = Field(..., description="List of completion choices")

# Initialize FastAPI app
app = FastAPI(title="LLaDA Diffusion API", version="1.0.0")

def generate_id() -> str:
    """Generate a unique ID for the completion"""
    return f"chatcmpl-{uuid.uuid4().hex[:32]}"

def format_messages_for_model(messages: List[Message]) -> List[Dict[str, str]]:
    """Convert API messages to model format"""
    return [{"role": msg.role, "content": msg.content} for msg in messages]

def generate_streaming_response_with_diffusion(
    model, tokenizer, device, messages, gen_length=64, steps=32, 
    constraints=None, temperature=0.0, block_length=32,
    remasking='low_confidence', threshold=0.9
) -> Generator[Dict[str, Any], None, None]:
    """
    Generate streaming response with diffusion model, yielding position-based updates
    Modified from generate_response_with_visualization_cache_and_parallel to yield streaming data
    """
    
    # Process constraints
    if constraints is None:
        constraints = {}
        
    # Convert any string constraints to token IDs
    processed_constraints = {}
    for pos, word in constraints.items():
        tokens = tokenizer.encode(" " + word, add_special_tokens=False)
        for i, token_id in enumerate(tokens):
            processed_constraints[pos + i] = token_id
    
    # Prepare the prompt using chat template
    chat_input = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer(chat_input)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
    
    # For generation
    prompt_length = input_ids.shape[1]
    
    # Initialize the sequence with masks for the response part
    x = torch.full((1, prompt_length + gen_length), MASK_ID, dtype=torch.long).to(device)
    x[:, :prompt_length] = input_ids.clone()
    
    # Apply constraints to the initial state
    for pos, token_id in processed_constraints.items():
        absolute_pos = prompt_length + pos
        if absolute_pos < x.shape[1]:
            x[:, absolute_pos] = token_id
    
    # Ensure block_length is valid
    if block_length > gen_length:
        block_length = gen_length
    
    # Calculate number of blocks
    num_blocks = gen_length // block_length
    if gen_length % block_length != 0:
        num_blocks += 1
    
    # Adjust steps per block
    steps_per_block = steps // num_blocks
    if steps_per_block < 1:
        steps_per_block = 1
    
    current_step = 0
    previous_state = {}  # Track previous token states
    
    # Process each block
    for num_block in range(num_blocks):
        current_block_start = prompt_length + num_block * block_length
        current_block_end = current_block_start + block_length

        block_mask_index = (x[:, current_block_start:current_block_end] == MASK_ID)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        output = model(x, use_cache=True)
        past_key_values = output.past_key_values

        mask_index = (x == MASK_ID)
        mask_index[:, current_block_end:] = 0
        x0, transfer_index = get_transfer_index(
            output.logits, temperature, remasking, mask_index, x, 
            num_transfer_tokens[:, 0] if threshold is None else None, threshold
        )
        x[transfer_index] = x0[transfer_index]

        new_past_key_values = []
        for i in range(len(past_key_values)):
            new_past_key_values.append(())
            for j in range(len(past_key_values[i])):
                new_past_key_values[i] += (past_key_values[i][j][:, :, :current_block_start],)
        
        past_key_values = new_past_key_values
        
        # Yield changes for this step
        current_step += 1
        change_data = yield_diffusion_changes(x, prompt_length, gen_length, previous_state, current_step, steps, tokenizer)
        if change_data:
            yield change_data
        
        i = 1
        while True:
            mask_index = (x[:, current_block_start:] == MASK_ID)
            mask_index[:, block_length:] = 0

            logits = model(x[:, current_block_start:], past_key_values=past_key_values, use_cache=True).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            x0, transfer_index = get_transfer_index(
                logits, temperature, remasking, mask_index, 
                x[:, current_block_start:], 
                num_transfer_tokens[:, i] if threshold is None else None, threshold
            )
            x[:, current_block_start:][transfer_index] = x0[transfer_index]
            
            # Yield changes for this step
            current_step += 1
            change_data = yield_diffusion_changes(x, prompt_length, gen_length, previous_state, current_step, steps, tokenizer)
            if change_data:
                yield change_data
            
            if (x[:, current_block_start:current_block_end] == MASK_ID).sum() == 0:
                break
            i += 1

def yield_diffusion_changes(x, prompt_length, gen_length, previous_state, current_step, max_steps, tokenizer):
    """Yield only the changed tokens in diffusion format"""
    changes = []
    
    for i in range(gen_length):
        pos = prompt_length + i
        if pos < x.shape[1]:
            current_token_id = x[0, pos].item()
            
            if current_token_id != MASK_ID:
                # Check if this position changed
                if i not in previous_state or previous_state[i] != current_token_id:
                    token_text = tokenizer.decode([current_token_id], skip_special_tokens=True)
                    changes.append(DiffusionContent(position=i, text=token_text))
                    previous_state[i] = current_token_id
    
    if changes:
        print(f"Step {current_step}/{max_steps}: {len(changes)} token changes")
        # for change in changes:
        #     print(f"  Position {change.position}: '{change.text}'")
        
        return {
            "content": [change.dict() for change in changes],
            "step": current_step,
            "max_step": max_steps
        }
    return None

async def stream_chat_completion(request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
    """Generate streaming chat completion response"""
    try:
        print(f"=== Starting Stream Generation ===")
        total_start_time = time.time()
        
        completion_id = generate_id()
        created = int(time.time())
        
        # Extract parameters
        messages = format_messages_for_model(request.messages)
        gen_length = request.max_tokens
        temperature = request.temperature
        
        # Extract diffusion parameters
        extra = request.extra_body or ExtraBody()
        steps = extra.diffusion_steps
        remasking = extra.diffusion_remasking
        if remasking not in ["low_confidence", "random"]:
            remasking = "low_confidence"
        block_length = extra.diffusion_block_length
        threshold = extra.diffusion_threshold
        
        print(f"Stream parameters: gen_length={gen_length}, steps={steps}, temperature={temperature}")
        print(f"Diffusion params: remasking={remasking}, block_length={block_length}, threshold={threshold}")
        
        # Measure preprocessing time
        prep_start = time.time()
        
        # Generate response with streaming
        previous_state = {}
        current_step = 0
        generation_start_time = time.time()
        
        prep_time = generation_start_time - prep_start
        print(f"Preprocessing time: {prep_time:.3f}s")
        
        first_token_time = None
        
        # Use the modified generation function that yields streaming data
        for step_data in generate_streaming_response_with_diffusion(
            model, tokenizer, device, messages, 
            gen_length=gen_length, steps=steps,
            temperature=temperature, block_length=block_length,
            remasking=remasking, threshold=threshold
        ):
            if step_data:
                if first_token_time is None:
                    first_token_time = time.time() - generation_start_time
                    print(f"Time to first token: {first_token_time:.3f}s")
                
                current_step += 1
                
                response = ChatCompletionResponse(
                    id=completion_id,
                    object="chat.completion.chunk",
                    created=created,
                    model=request.model,
                    choices=[Choice(
                        index=0,
                        delta=DiffusionDelta(**step_data),
                        finish_reason=None
                    )]
                )
                
                chunk_data = f"data: {response.model_dump_json()}\n\n"
                print(f"Streaming chunk (step {step_data['step']}/{step_data['max_step']}): {len(step_data['content'])} changes")
                yield chunk_data
        
        total_generation_time = time.time() - generation_start_time
        total_time = time.time() - total_start_time
        
        print(f"=== Stream Generation Complete ===")
        print(f"Total request time: {total_time:.2f}s")
        print(f"Generation time: {total_generation_time:.2f}s")
        print(f"Time to first token: {first_token_time:.3f}s" if first_token_time else "No tokens generated")
        print(f"Total steps: {current_step}")
        if current_step > 0:
            print(f"Average time per step: {total_generation_time/current_step:.3f}s")
        print(f"===================================")
        
        # Send final completion message
        final_response = ChatCompletionResponse(
            id=completion_id,
            object="chat.completion.chunk",
            created=created,
            model=request.model,
            choices=[Choice(
                index=0,
                delta=DiffusionDelta(content=[], step=current_step, max_step=steps),
                finish_reason="stop"
            )]
        )
        
        yield f"data: {final_response.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        print(f"Error in stream_chat_completion: {e}")
        print(traceback.format_exc())
        error_response = {
            "error": {
                "message": str(e),
                "type": "internal_error",
                "code": "internal_error"
            }
        }
        yield f"data: {json.dumps(error_response)}\n\n"

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint"""
    try:
        request_start_time = time.time()
        
        # Print incoming request
        print(f"\n=== Incoming Request ===")
        print(f"Model: {request.model}")
        print(f"Stream: {request.stream}")
        print(f"Max tokens: {request.max_tokens}")
        print(f"Temperature: {request.temperature}")
        print(f"Messages: {json.dumps([msg.dict() for msg in request.messages], indent=2, ensure_ascii=False)}")
        if request.extra_body:
            print(f"Extra body: {request.extra_body.dict()}")
        print(f"========================\n")

        print(f"v1/chat/completions: {request}")

        # Measure model loading time
        load_start = time.time()
        load_model()  # Ensure model is loaded
        load_time = time.time() - load_start
        
        if load_time > 0.1:  # Only print if significant loading time
            print(f"Model loading/check time: {load_time:.3f}s")
        
        if request.stream:
            print(f"=== Starting Streaming Response ===")
            return StreamingResponse(
                stream_chat_completion(request),
                media_type="text/plain",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Content-Type": "text/plain; charset=utf-8"
                }
            )
        else:
            # Non-streaming response
            completion_id = generate_id()
            created = int(time.time())
            
            # Extract parameters
            messages = format_messages_for_model(request.messages)
            gen_length = request.max_tokens
            temperature = request.temperature
            
            # Extract diffusion parameters
            extra = request.extra_body or ExtraBody()
            steps = extra.diffusion_steps
            remasking = extra.diffusion_remasking
            if remasking not in ["low_confidence", "random"]:
                remasking = "low_confidence"
            block_length = extra.diffusion_block_length
            threshold = extra.diffusion_threshold
            
            # Generate complete response
            print(f"=== Generating Non-streaming Response ===")
            generation_start = time.time()
            _, final_text = generate_response_with_visualization_cache_and_parallel(
                model, tokenizer, device, messages,
                gen_length=gen_length, steps=steps,
                temperature=temperature, block_length=block_length,
                remasking=remasking, threshold=threshold
            )
            generation_time = time.time() - generation_start
            total_request_time = time.time() - request_start_time
            
            print(f"=== Generation Complete ===")
            print(f"Total request time: {total_request_time:.2f}s")
            print(f"Generation time: {generation_time:.2f}s")
            print(f"Overhead time: {total_request_time - generation_time:.3f}s")
            print(f"Generated text: {final_text}")
            print(f"==========================\n")
            
            response = ChatCompletionResponse(
                id=completion_id,
                object="chat.completion",
                created=created,
                model=request.model,
                choices=[Choice(
                    index=0,
                    message={"role": "assistant", "content": final_text},
                    finish_reason="stop"
                )]
            )
            
            print(f"=== Final Response ===")
            print(f"Response: {response.model_dump_json()}")
            print(f"======================\n")
            
            return response
            
    except Exception as e:
        print(f"Error in chat_completions: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/models")
async def list_models():
    """List available models"""
    return {
        "object": "list",
        "data": [
            {
                "id": "llada-8b-instruct",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "llada",
                "permission": [],
                "root": "llada-8b-instruct",
                "parent": None
            }
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None}

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 50)
    print("Starting LLaDA Diffusion API Server")
    print("=" * 50)
    
    # Pre-load model on startup to avoid first request delay
    print("Pre-loading model to avoid first request delay...")
    startup_time = time.time()
    load_model()
    startup_duration = time.time() - startup_time
    print(f"Startup completed in {startup_duration:.2f}s")
    print("=" * 50)
    
    uvicorn.run(app, host="0.0.0.0", port=18081)