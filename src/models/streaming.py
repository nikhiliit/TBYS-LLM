"""Streaming token generation for Qwen models."""

import json
from typing import List, Optional, Generator, Dict, Any

import torch

from .qwen_manager import Qwen3Manager
from ..database import db_manager
class StreamingGenerator:
    """Helper class for streaming token generation."""

    def __init__(self, manager: Qwen3Manager, prompt: str, conversation_id: int,
                 enable_thinking: bool = True, max_new_tokens: int = 2048,
                 temperature: float = 0.6, pdf_images: Optional[List] = None):
        self.manager = manager
        self.prompt = prompt
        self.conversation_id = conversation_id
        self.enable_thinking = enable_thinking
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.pdf_images = pdf_images or []
        self.thinking_token_id = 151668  # </think>
        
    def generate_stream(self) -> Generator[str, None, None]:
        """Stream tokens as they're generated."""
        if not self.manager.model or not self.manager.tokenizer:
            yield f"data: {json.dumps({'error': 'Model not loaded'})}\n\n"
            return
        
        try:
            # Prepare messages with conversation history
            messages = self._get_conversation_messages()
            messages.append({"role": "user", "content": self.prompt})
            
            # Apply chat template
            text = self.manager.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking
            )

            # Debug: Check if <think> tag is in the prompt
            if self.enable_thinking:
                if '<think>' in text:
                    print("[DEBUG] ✅ <think> tag found in prompt")
                else:
                    print("[DEBUG] ⚠️  <think> tag NOT found in prompt! This happens with very long prompts.")
                    print("[DEBUG] Solution: Truncate PDF content or disable thinking for long documents")
                    # Note: We don't manually add tags as it may break tokenization
                    # Instead, we'll still try to detect thinking tokens in the output

            # Tokenize
            model_inputs = self.manager.tokenizer([text], return_tensors="pt")
            model_inputs = {k: v.to(self.manager.device) for k, v in model_inputs.items()}
            
            input_length = len(model_inputs['input_ids'][0])
            
            # Generate with streaming
            thinking_buffer = []
            response_buffer = []
            in_thinking = True
            top_p = 0.95 if self.enable_thinking else 0.8
            top_k = 20

            print(f"[DEBUG] Starting generation: enable_thinking={self.enable_thinking}, in_thinking={in_thinking}")
            print(f"[DEBUG] Thinking end token ID: {self.thinking_token_id}")
            print(f"[DEBUG] Prompt length: {len(self.prompt)} chars")
            
            with torch.no_grad():
                # Use manual token-by-token processing for streaming
                generated_ids = model_inputs['input_ids'].clone()
                
                for step in range(self.max_new_tokens):
                    # Get next token logits
                    outputs = self.manager.model(generated_ids)
                    logits = outputs.logits[:, -1, :]

                    # Safety check: handle inf/nan logits from model
                    if torch.isnan(logits).any() or torch.isinf(logits).any():
                        print(f"[DEBUG] Invalid logits detected, clamping to safe range")
                        logits = torch.clamp(logits, min=-100, max=100)

                    # Apply temperature
                    if self.temperature > 0 and self.temperature != 1.0:
                        logits = logits / self.temperature
                    
                    # Apply top-k filtering
                    if top_k > 0:
                        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                        logits[indices_to_remove] = float('-inf')

                    # Apply top-p (nucleus) filtering
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                        # Remove tokens with cumulative probability above the threshold
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0

                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        logits[indices_to_remove] = float('-inf')

                    # Safety check: ensure at least one logit is not -inf
                    if torch.all(logits == float('-inf')):
                        # Reset to original logits if all were filtered out
                        outputs = self.manager.model(generated_ids)
                        logits = outputs.logits[:, -1, :]
                        if self.temperature > 0 and self.temperature != 1.0:
                            logits = logits / self.temperature
                    
                    # Sample next token
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    
                    # Append to generated_ids
                    generated_ids = torch.cat([generated_ids, next_token], dim=1)
                    
                    token_id = next_token.item()

                    # Debug: Log first 10 tokens to understand what's being generated
                    if step < 10:
                        decoded = self.manager.tokenizer.decode([token_id])
                        print(f"[DEBUG] Token {step}: id={token_id}, text='{decoded}', in_thinking={in_thinking}")

                    # Check for EOS
                    if token_id == self.manager.tokenizer.eos_token_id:
                        print(f"[DEBUG] EOS token reached at step {step}, thinking_buffer_len={len(thinking_buffer)}")
                        yield f"data: {json.dumps({'type': 'done'})}\n\n"
                        break

                    # Check for thinking end token (</think>)
                    if token_id == self.thinking_token_id and in_thinking:
                        in_thinking = False
                        thinking_text = self.manager.tokenizer.decode(
                            thinking_buffer, skip_special_tokens=True
                        )
                        print(f"[DEBUG] Thinking complete! Tokens: {len(thinking_buffer)}, Text length: {len(thinking_text)}")
                        yield f"data: {json.dumps({'type': 'thinking_complete', 'content': thinking_text})}\n\n"
                        continue
                    
                    # Decode and stream token
                    if in_thinking:
                        thinking_buffer.append(token_id)
                        # Decode accumulated thinking so far for better display
                        if len(thinking_buffer) % 5 == 0 or step == 0:  # Decode every 5 tokens for efficiency
                            thinking_text = self.manager.tokenizer.decode(
                                thinking_buffer, skip_special_tokens=True
                            )
                            print(f"[DEBUG] Streaming thinking token (step {step}): len={len(thinking_buffer)}")
                            yield f"data: {json.dumps({'type': 'thinking', 'content': thinking_text})}\n\n"
                    else:
                        response_buffer.append(token_id)
                        # Decode accumulated response so far
                        if len(response_buffer) % 3 == 0 or step == 0:  # Decode every 3 tokens
                            response_text = self.manager.tokenizer.decode(
                                response_buffer, skip_special_tokens=True
                            )
                            yield f"data: {json.dumps({'type': 'response', 'content': response_text})}\n\n"
            
            # Finalize - decode remaining tokens
            if in_thinking and thinking_buffer:
                thinking_text = self.manager.tokenizer.decode(
                    thinking_buffer, skip_special_tokens=True
                )
                yield f"data: {json.dumps({'type': 'thinking_complete', 'content': thinking_text})}\n\n"
            
            if response_buffer:
                response_text = self.manager.tokenizer.decode(
                    response_buffer, skip_special_tokens=True
                )
                yield f"data: {json.dumps({'type': 'response_complete', 'content': response_text})}\n\n"
            
            # Save to database
            final_thinking = self.manager.tokenizer.decode(thinking_buffer, skip_special_tokens=True) if thinking_buffer else ""
            final_response = self.manager.tokenizer.decode(response_buffer, skip_special_tokens=True) if response_buffer else ""
            self._save_message(final_thinking, final_response)
            
        except Exception as e:
            import traceback
            yield f"data: {json.dumps({'error': str(e), 'traceback': traceback.format_exc()})}\n\n"
    
    def _get_conversation_messages(self) -> List[Dict[str, str]]:
        """Get conversation history from database."""
        if self.conversation_id <= 0:
            return []

        # Get conversation data using db_manager
        conversation = db_manager.get_conversation(self.conversation_id)
        if not conversation or 'messages' not in conversation:
            return []

        # Convert to the format expected by the chat template
        messages = []
        for msg in conversation['messages']:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        return messages
    
    def _save_message(self, thinking: str, response: str):
        """Save assistant response to database (user message already saved)."""
        if self.conversation_id <= 0:
            return

        # Save assistant response using db_manager
        content = response
        if thinking:
            content = f"{thinking}\n\n{response}" if response else thinking

        db_manager.save_message(self.conversation_id, "assistant", content)