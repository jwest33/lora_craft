"""Model testing module for comparing trained models with base models."""

import os
import json
import torch
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from datetime import datetime
import gc

# Disable torch.compile/dynamo to avoid FX tracing issues
os.environ["TORCHDYNAMO_DISABLE"] = "1"

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextStreamer,
    GenerationConfig
)
from peft import PeftModel

from core.reward_loader import load_reward_from_session

logger = logging.getLogger(__name__)


@dataclass
class TestConfig:
    """Configuration for model testing."""
    temperature: float = 0.7
    max_new_tokens: int = 512
    top_p: float = 0.95
    top_k: int = 50
    repetition_penalty: float = 1  # Increased from 1.0 to reduce repetition
    do_sample: bool = True
    num_beams: int = 1
    use_cache: bool = True

    def to_generation_config(self) -> GenerationConfig:
        """Convert to HuggingFace GenerationConfig."""
        return GenerationConfig(
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            do_sample=self.do_sample,
            num_beams=self.num_beams,
            use_cache=self.use_cache,
            pad_token_id=2,  # Default for most models
            eos_token_id=2
        )


class ModelTester:
    """Handle model testing and comparison."""

    def __init__(self):
        """Initialize the ModelTester."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.loaded_models = {}
        self.base_chat_template = """<|system|>
You are a helpful AI assistant.
<|user|>
{prompt}
<|assistant|>"""

    def load_trained_model(
        self,
        checkpoint_path: str,
        session_id: str
    ) -> Tuple[bool, Optional[str]]:
        """Load a trained model from checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint
            session_id: Session ID for caching

        Returns:
            Tuple of (success, error_message)
        """
        try:
            cache_key = f"trained_{session_id}"

            # Check if already loaded
            if cache_key in self.loaded_models:
                logger.info(f"Using cached trained model for {session_id}")
                return True, None

            checkpoint_path = Path(checkpoint_path)
            if not checkpoint_path.exists():
                return False, f"Checkpoint not found: {checkpoint_path}"

            logger.info(f"Loading trained model from {checkpoint_path}")

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Ensure padding side is set correctly for generation
            tokenizer.padding_side = 'left'

            # Check if it's a LoRA model
            adapter_config_path = checkpoint_path / "adapter_config.json"
            if adapter_config_path.exists():
                # Load base model and LoRA adapter
                with open(adapter_config_path, 'r') as f:
                    adapter_config = json.load(f)
                base_model_name = adapter_config.get("base_model_name_or_path")

                logger.info(f"Loading base model: {base_model_name}")
                # Always use float16 for consistency with Unsloth training
                model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    low_cpu_mem_usage=True
                )

                logger.info("Loading LoRA adapter")
                model = PeftModel.from_pretrained(model, checkpoint_path)

                # Ensure the model is in float16 if on CUDA
                if self.device == "cuda":
                    # Convert entire model to float16
                    model = model.to(torch.float16)
                    logger.info("Converted model to float16 for consistency")

                    # Ensure all submodules are also in float16
                    for name, module in model.named_modules():
                        for param_name, param in module.named_parameters(recurse=False):
                            if param.dtype != torch.float16:
                                param.data = param.data.to(torch.float16)
                                logger.debug(f"Converted {name}.{param_name} to float16")
            else:
                # Load full model
                model = AutoModelForCausalLM.from_pretrained(
                    checkpoint_path,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    low_cpu_mem_usage=True
                )

                # Ensure the model is in float16 if on CUDA
                if self.device == "cuda":
                    # Convert entire model to float16
                    model = model.to(torch.float16)
                    logger.info("Converted full model to float16 for consistency")

                    # Ensure all submodules are also in float16
                    for name, module in model.named_modules():
                        for param_name, param in module.named_parameters(recurse=False):
                            if param.dtype != torch.float16:
                                param.data = param.data.to(torch.float16)
                                logger.debug(f"Converted {name}.{param_name} to float16")

            model.eval()

            # Cache the model
            self.loaded_models[cache_key] = {
                "model": model,
                "tokenizer": tokenizer,
                "loaded_at": datetime.now().isoformat()
            }

            logger.info(f"Successfully loaded trained model for {session_id}")
            return True, None

        except Exception as e:
            logger.error(f"Failed to load trained model: {str(e)}")
            return False, str(e)

    def load_base_model(
        self,
        model_name: str
    ) -> Tuple[bool, Optional[str]]:
        """Load a base model for comparison.

        Args:
            model_name: Name of the base model

        Returns:
            Tuple of (success, error_message)
        """
        try:
            cache_key = f"base_{model_name.replace('/', '_')}"

            # Check if already loaded
            if cache_key in self.loaded_models:
                logger.info(f"Using cached base model {model_name}")
                return True, None

            logger.info(f"Loading base model: {model_name}")

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Load model with explicit configuration to avoid optimization issues
            model_kwargs = {
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                "device_map": "auto" if self.device == "cuda" else None,
                "low_cpu_mem_usage": True,
                "use_cache": True,  # Explicitly enable cache
                "trust_remote_code": False,  # Avoid custom optimized code
                "attn_implementation": "eager"  # Use eager mode instead of optimized implementations
            }

            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    **model_kwargs
                )
            except Exception as e:
                # Fallback without attn_implementation if not supported
                logger.warning(f"Failed with attn_implementation=eager, retrying without: {e}")
                del model_kwargs["attn_implementation"]
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    **model_kwargs
                )

            # Ensure consistent dtype for CUDA
            if self.device == "cuda":
                # Convert entire model to float16
                model = model.to(torch.float16)
                logger.info("Converted base model to float16")

                # Ensure all submodules are also in float16
                for name, module in model.named_modules():
                    for param_name, param in module.named_parameters(recurse=False):
                        if param.dtype != torch.float16:
                            param.data = param.data.to(torch.float16)
                            logger.debug(f"Converted {name}.{param_name} to float16")

            model.eval()

            # Cache the model
            self.loaded_models[cache_key] = {
                "model": model,
                "tokenizer": tokenizer,
                "loaded_at": datetime.now().isoformat()
            }

            logger.info(f"Successfully loaded base model {model_name}")
            return True, None

        except Exception as e:
            logger.error(f"Failed to load base model: {str(e)}")
            return False, str(e)

    def evaluate_response_with_reward(
        self,
        instruction: str,
        generated: str,
        reference: Optional[str] = None,
        session_info = None,
        tokenizer = None
    ) -> Dict[str, Any]:
        """Evaluate a generated response using the session's reward function.

        Args:
            instruction: Input instruction/prompt
            generated: Generated response to evaluate
            reference: Reference response for comparison (optional)
            session_info: SessionInfo object with reward_config
            tokenizer: Optional tokenizer for token-based metrics

        Returns:
            Dictionary with:
                - success: Whether evaluation succeeded
                - total_reward: Overall reward score (0.0-1.0)
                - components: Breakdown of component scores
                - error: Error message if evaluation failed
        """
        try:
            # Load reward function from session
            reward_builder = load_reward_from_session(session_info)

            if not reward_builder:
                return {
                    "success": False,
                    "error": "No reward configuration found in session",
                    "total_reward": None,
                    "components": {}
                }

            # Compute reward
            total_reward, component_rewards = reward_builder.compute_total_reward(
                instruction=instruction,
                generated=generated,
                reference=reference,
                tokenizer=tokenizer
            )

            return {
                "success": True,
                "total_reward": total_reward,
                "components": component_rewards,
                "error": None
            }

        except Exception as e:
            logger.error(f"Failed to evaluate response with reward: {e}")
            return {
                "success": False,
                "error": str(e),
                "total_reward": None,
                "components": {}
            }

    def generate_response(
        self,
        prompt: str,
        model_type: str,
        model_key: str,
        config: Optional[TestConfig] = None,
        use_chat_template: bool = True,
        streaming_callback = None,
        use_simple_prompt: bool = False,
        session_info = None,
        override_system_prompt: Optional[str] = None,
        evaluate_reward: bool = False,
        reference_response: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate response from a model.

        Args:
            prompt: Input prompt
            model_type: "trained" or "base"
            model_key: Session ID for trained or model name for base
            config: Generation configuration
            use_chat_template: Whether to apply chat template
            streaming_callback: Callback for streaming tokens
            use_simple_prompt: Use simple prompting without system instructions
            session_info: SessionInfo object with training configuration (for trained models)
            override_system_prompt: Custom system prompt to override session config
            evaluate_reward: Whether to evaluate the response using session's reward function
            reference_response: Reference response for reward comparison (optional)

        Returns:
            Dictionary with response and metadata (includes reward data if evaluate_reward=True)
        """
        try:
            # Get the model key
            if model_type == "trained":
                cache_key = f"trained_{model_key}"
            else:
                cache_key = f"base_{model_key.replace('/', '_')}"

            if cache_key not in self.loaded_models:
                return {
                    "success": False,
                    "error": f"Model not loaded: {model_key}"
                }

            model_data = self.loaded_models[cache_key]
            model = model_data["model"]
            tokenizer = model_data["tokenizer"]

            # Use default config if not provided
            if config is None:
                config = TestConfig()

            # Track system prompt for metadata
            system_prompt = None

            # Apply chat template if requested
            if use_simple_prompt:
                # For simple testing, just use the prompt directly
                formatted_prompt = prompt
                logger.debug("Using simple prompt format for testing")
            elif use_chat_template:
                # Try to use chat template from training configuration if available
                logger.debug(f"use_chat_template=True, session_info exists: {session_info is not None}")
                if session_info:
                    logger.debug(f"session_info has training_config: {hasattr(session_info, 'training_config')}")
                    if hasattr(session_info, 'training_config'):
                        logger.debug(f"training_config is None: {session_info.training_config is None}")

                if session_info and hasattr(session_info, 'training_config') and session_info.training_config:
                    training_config = session_info.training_config
                    logger.debug(f"Using training_config from session_info")

                    # Get chat template from training config
                    chat_template = (
                        training_config.get('chat_template') or
                        training_config.get('template', {}).get('chat_template')
                    )

                    # Get system prompt (use override if provided, otherwise from config)
                    system_prompt = override_system_prompt or (
                        training_config.get('system_prompt') or
                        training_config.get('template', {}).get('system_prompt') or
                        'You are a helpful AI assistant.'
                    )
                    logger.debug(f"Extracted system_prompt: {system_prompt[:100] if system_prompt else None}")

                    if chat_template:
                        # Use the training chat template
                        try:
                            from jinja2 import Environment
                            env = Environment()
                            template = env.from_string(chat_template)

                            # Prepare messages
                            messages = []
                            if system_prompt:
                                messages.append({"role": "system", "content": system_prompt})
                            messages.append({"role": "user", "content": prompt})

                            # Get template markers from config (check both root and template object)
                            reasoning_start = (
                                training_config.get('reasoning_start') or
                                training_config.get('template', {}).get('reasoning_start') or
                                '<start_working_out>'
                            )
                            eos_token = '</s>'  # Common default

                            # Render template
                            formatted_prompt = template.render(
                                messages=messages,
                                add_generation_prompt=True,
                                eos_token=eos_token,
                                system_prompt=system_prompt,
                                reasoning_start=reasoning_start
                            )
                            logger.debug(f"Applied chat template from training config")
                        except Exception as e:
                            logger.warning(f"Failed to apply training chat template: {e}")
                            # Fallback to tokenizer's template
                            if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
                                try:
                                    messages = [{"role": "user", "content": prompt}]
                                    formatted_prompt = tokenizer.apply_chat_template(
                                        messages,
                                        tokenize=False,
                                        add_generation_prompt=True
                                    )
                                    logger.debug(f"Applied tokenizer chat template as fallback")
                                except Exception as e2:
                                    logger.warning(f"Failed to apply tokenizer chat template: {e2}")
                                    formatted_prompt = prompt
                            else:
                                formatted_prompt = prompt
                    else:
                        # Use chat template type to format
                        chat_template_type = (
                            training_config.get('chat_template_type') or
                            training_config.get('template', {}).get('chat_template_type') or
                            'grpo'
                        )
                        # Get reasoning_start from config (check both root and template object)
                        reasoning_start = (
                            training_config.get('reasoning_start') or
                            training_config.get('template', {}).get('reasoning_start') or
                            '<start_working_out>'
                        )

                        if chat_template_type == 'grpo':
                            # GRPO format
                            if system_prompt:
                                formatted_prompt = f"{system_prompt}</s>{prompt}{reasoning_start}"
                            else:
                                formatted_prompt = f"{prompt}{reasoning_start}"
                        elif chat_template_type == 'qwen':
                            # Qwen format
                            if system_prompt:
                                formatted_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
                            else:
                                formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
                        elif chat_template_type == 'llama':
                            # LLaMA format
                            if system_prompt:
                                formatted_prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt} [/INST]"
                            else:
                                formatted_prompt = f"<s>[INST] {prompt} [/INST]"
                        else:
                            # Generic format
                            if system_prompt:
                                formatted_prompt = f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"
                            else:
                                formatted_prompt = f"User: {prompt}\n\nAssistant:"

                        logger.debug(f"Applied {chat_template_type} chat template from training config")
                elif model_type == "base":
                    # Use training system prompt if session_info provided, otherwise generic
                    if session_info and hasattr(session_info, 'training_config') and session_info.training_config:
                        training_config = session_info.training_config

                        # Get system prompt from training config
                        system_prompt = override_system_prompt or (
                            training_config.get('system_prompt') or
                            training_config.get('template', {}).get('system_prompt') or
                            'You are a helpful AI assistant.'
                        )

                        # Get chat template configuration
                        chat_template = (
                            training_config.get('chat_template') or
                            training_config.get('template', {}).get('chat_template')
                        )

                        if chat_template:
                            # Use the training chat template
                            try:
                                from jinja2 import Environment
                                env = Environment()
                                template = env.from_string(chat_template)

                                # Prepare messages
                                messages = []
                                if system_prompt:
                                    messages.append({"role": "system", "content": system_prompt})
                                messages.append({"role": "user", "content": prompt})

                                # Get template markers from config (check both root and template object)
                                reasoning_start = (
                                    training_config.get('reasoning_start') or
                                    training_config.get('template', {}).get('reasoning_start') or
                                    '<start_working_out>'
                                )
                                eos_token = '</s>'  # Common default

                                # Render template
                                formatted_prompt = template.render(
                                    messages=messages,
                                    add_generation_prompt=True,
                                    eos_token=eos_token,
                                    system_prompt=system_prompt,
                                    reasoning_start=reasoning_start
                                )
                                logger.debug(f"Applied training chat template to base model")
                            except Exception as e:
                                logger.warning(f"Failed to apply training chat template to base model: {e}")
                                # Fallback to formatted prompt with system prompt
                                if system_prompt:
                                    formatted_prompt = f"{system_prompt}</s>{prompt}"
                                else:
                                    formatted_prompt = prompt
                        else:
                            # Use chat template type to format
                            chat_template_type = (
                                training_config.get('chat_template_type') or
                                training_config.get('template', {}).get('chat_template_type') or
                                'grpo'
                            )
                            # Get reasoning_start from config (check both root and template object)
                            reasoning_start = (
                                training_config.get('reasoning_start') or
                                training_config.get('template', {}).get('reasoning_start') or
                                '<start_working_out>'
                            )

                            if chat_template_type == 'grpo':
                                # GRPO format
                                if system_prompt:
                                    formatted_prompt = f"{system_prompt}</s>{prompt}{reasoning_start}"
                                else:
                                    formatted_prompt = f"{prompt}{reasoning_start}"
                            elif chat_template_type == 'qwen':
                                # Qwen format
                                if system_prompt:
                                    formatted_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
                                else:
                                    formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
                            elif chat_template_type == 'llama':
                                # LLaMA format
                                if system_prompt:
                                    formatted_prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt} [/INST]"
                                else:
                                    formatted_prompt = f"<s>[INST] {prompt} [/INST]"
                            else:
                                # Generic format
                                if system_prompt:
                                    formatted_prompt = f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"
                                else:
                                    formatted_prompt = f"User: {prompt}\n\nAssistant:"

                            logger.debug(f"Applied {chat_template_type} chat template from training config to base model")
                    else:
                        # Use generic chat template for base model without training config
                        system_prompt = "You are a helpful AI assistant."
                        formatted_prompt = self.base_chat_template.format(prompt=prompt)
                        logger.debug("Applied generic base chat template")
                else:
                    # Try to use model's tokenizer chat template if available
                    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
                        try:
                            messages = [{"role": "user", "content": prompt}]
                            formatted_prompt = tokenizer.apply_chat_template(
                                messages,
                                tokenize=False,
                                add_generation_prompt=True
                            )
                            logger.debug(f"Applied tokenizer chat template")
                        except Exception as e:
                            logger.warning(f"Failed to apply tokenizer chat template: {e}")
                            formatted_prompt = prompt
                    else:
                        logger.warning("No chat template found, using prompt as-is")
                        formatted_prompt = prompt
            else:
                formatted_prompt = prompt

            # Tokenize input
            inputs = tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            )

            # Move inputs to the correct device
            if self.device == "cuda":
                # Move inputs to CUDA
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

                # Ensure model is in evaluation mode
                model.eval()

                # Get the model's dtype for consistency (should be float16)
                model_dtype = next(model.parameters()).dtype
                logger.debug(f"Model dtype: {model_dtype}")

                # Ensure inputs match model dtype
                for key in inputs.keys():
                    if key == 'input_ids':
                        # input_ids should remain as long
                        if inputs[key].dtype != torch.long:
                            inputs[key] = inputs[key].long()
                    elif key == 'attention_mask':
                        # attention_mask should match model dtype
                        if inputs[key].dtype != model_dtype:
                            inputs[key] = inputs[key].to(model_dtype)
                            logger.debug(f"Converted attention_mask to {model_dtype}")
                    else:
                        # Any other tensor should match model dtype
                        if inputs[key].dtype != model_dtype:
                            inputs[key] = inputs[key].to(model_dtype)
                            logger.debug(f"Converted {key} to {model_dtype}")

            # Set up streaming if callback provided
            streamer = None
            if streaming_callback:
                class CustomStreamer(TextStreamer):
                    def __init__(self, tokenizer, callback):
                        super().__init__(tokenizer, skip_prompt=True)
                        self.callback = callback
                        self.tokens = []

                    def on_finalized_text(self, text, stream_end=False):
                        self.tokens.append(text)
                        self.callback(text, stream_end)

                streamer = CustomStreamer(tokenizer, streaming_callback)

            # Generate response with fallback for FX tracing errors
            start_time = datetime.now()
            outputs = None
            generation_error = None

            try:
                with torch.no_grad():
                    generation_config = config.to_generation_config()
                    outputs = model.generate(
                        **inputs,
                        generation_config=generation_config,
                        streamer=streamer
                    )
            except Exception as e:
                if "FX" in str(e) or "dynamo" in str(e) or "symbolically trace" in str(e):
                    logger.warning(f"FX tracing error detected: {e}")
                    logger.info("Attempting generation with simplified config...")

                    # Try with simplified generation without advanced features
                    try:
                        with torch.no_grad():
                            # Use simpler generation parameters
                            outputs = model.generate(
                                **inputs,
                                max_new_tokens=config.max_new_tokens,
                                temperature=config.temperature,
                                do_sample=config.do_sample,
                                top_p=config.top_p,
                                pad_token_id=tokenizer.pad_token_id,
                                eos_token_id=tokenizer.eos_token_id,
                                use_cache=True
                            )
                    except Exception as e2:
                        logger.error(f"Failed even with simplified generation: {e2}")
                        generation_error = str(e2)
                else:
                    logger.error(f"Generation error: {e}")
                    generation_error = str(e)

            end_time = datetime.now()
            generation_time = (end_time - start_time).total_seconds()

            if outputs is None:
                # Return error if generation completely failed
                return {
                    "success": False,
                    "error": f"Generation failed: {generation_error}",
                    "metadata": {
                        "model_type": model_type,
                        "model_key": model_key
                    }
                }

            # Decode response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Remove the prompt from response
            if response.startswith(formatted_prompt):
                response = response[len(formatted_prompt):].strip()

            # Calculate token counts
            input_tokens = len(inputs['input_ids'][0])
            output_tokens = len(outputs[0]) - input_tokens

            # Calculate tokens per second
            tokens_per_second = output_tokens / generation_time if generation_time > 0 else 0

            result = {
                "success": True,
                "response": response,
                "metadata": {
                    "model_type": model_type,
                    "model_key": model_key,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                    "generation_time": generation_time,
                    "token_count": output_tokens,  # Add for frontend compatibility
                    "tokens_per_second": tokens_per_second,  # Add for frontend compatibility
                    "formatted_prompt": formatted_prompt,
                    "system_prompt": system_prompt,
                    "config": {
                        "temperature": config.temperature,
                        "max_new_tokens": config.max_new_tokens,
                        "top_p": config.top_p,
                        "top_k": config.top_k
                    }
                }
            }

            # Optionally evaluate with reward function
            if evaluate_reward and session_info:
                reward_eval = self.evaluate_response_with_reward(
                    instruction=prompt,
                    generated=response,
                    reference=reference_response,
                    session_info=session_info,
                    tokenizer=tokenizer
                )
                result["reward_evaluation"] = reward_eval

            return result

        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def test_with_reward(
        self,
        prompt: str,
        model_type: str,
        model_key: str,
        session_info,
        reference_response: Optional[str] = None,
        config: Optional[TestConfig] = None
    ) -> Dict[str, Any]:
        """Generate and evaluate a response with the session's reward function.

        This is a convenience method that combines generation and reward evaluation.

        Args:
            prompt: Input prompt
            model_type: "trained" or "base"
            model_key: Session ID for trained or model name for base
            session_info: SessionInfo object with reward configuration
            reference_response: Optional reference response for comparison
            config: Generation configuration

        Returns:
            Dictionary with generation and reward evaluation results
        """
        return self.generate_response(
            prompt=prompt,
            model_type=model_type,
            model_key=model_key,
            config=config,
            use_chat_template=True,
            session_info=session_info,
            evaluate_reward=True,
            reference_response=reference_response
        )

    def compare_two_models(
        self,
        prompt: str,
        model1_session_id: str,
        model2_session_id: str,
        config: Optional[TestConfig] = None,
        use_chat_template: bool = True,
        use_simple_prompt: bool = False,
        model1_session_info = None,
        model2_session_info = None
    ) -> Dict[str, Any]:
        """Compare responses from two trained models.

        Args:
            prompt: Input prompt
            model1_session_id: Session ID of first model
            model2_session_id: Session ID of second model
            config: Generation configuration
            use_chat_template: Whether to apply chat template
            use_simple_prompt: Use simple prompting without system instructions
            model1_session_info: SessionInfo object for first model
            model2_session_info: SessionInfo object for second model

        Returns:
            Dictionary with comparison results
        """
        results = {
            "prompt": prompt,
            "timestamp": datetime.now().isoformat(),
            "config": config.__dict__ if config else TestConfig().__dict__,
            "comparison_type": "model_vs_model"
        }

        # Generate from first model using its training chat template
        model1_result = self.generate_response(
            prompt=prompt,
            model_type="trained",
            model_key=model1_session_id,
            config=config,
            use_chat_template=use_chat_template,
            use_simple_prompt=use_simple_prompt,
            session_info=model1_session_info
        )
        results["model1"] = {
            **model1_result,
            "session_id": model1_session_id
        }

        # Generate from second model using its training chat template
        model2_result = self.generate_response(
            prompt=prompt,
            model_type="trained",
            model_key=model2_session_id,
            config=config,
            use_chat_template=use_chat_template,
            use_simple_prompt=use_simple_prompt,
            session_info=model2_session_info
        )
        results["model2"] = {
            **model2_result,
            "session_id": model2_session_id
        }

        # Calculate comparison metrics
        if model1_result["success"] and model2_result["success"]:
            results["metrics"] = self._calculate_comparison_metrics(
                model1_result["response"],
                model2_result["response"]
            )

        return results

    def _calculate_comparison_metrics(self, response1: str, response2: str) -> Dict[str, Any]:
        """Calculate comparison metrics between two model responses.

        Args:
            response1: First model's response
            response2: Second model's response

        Returns:
            Dictionary containing comparison metrics
        """
        return {
            "model1_length": len(response1),
            "model2_length": len(response2),
            "length_diff": len(response1) - len(response2),
            "model1_tokens": len(response1.split()),
            "model2_tokens": len(response2.split()),
            "token_diff": len(response1.split()) - len(response2.split())
        }

    def compare_models(
        self,
        prompt: str,
        trained_session_id: str,
        base_model_name: str,
        config: Optional[TestConfig] = None,
        use_chat_template: bool = True,
        use_simple_prompt: bool = False,
        session_info = None
    ) -> Dict[str, Any]:
        """Compare responses from trained and base models.

        Args:
            prompt: Input prompt
            trained_session_id: Session ID of trained model
            base_model_name: Name of base model
            config: Generation configuration
            use_chat_template: Whether to apply chat template
            use_simple_prompt: Use simple prompting without system instructions
            session_info: SessionInfo object for the trained model

        Returns:
            Dictionary with comparison results
        """
        results = {
            "prompt": prompt,
            "timestamp": datetime.now().isoformat(),
            "config": config.__dict__ if config else TestConfig().__dict__
        }

        # Generate from trained model with its training chat template
        trained_result = self.generate_response(
            prompt=prompt,
            model_type="trained",
            model_key=trained_session_id,
            config=config,
            use_chat_template=use_chat_template,
            use_simple_prompt=use_simple_prompt,
            session_info=session_info
        )
        results["trained"] = trained_result

        # Generate from base model using same system prompt as trained model
        base_result = self.generate_response(
            prompt=prompt,
            model_type="base",
            model_key=base_model_name,
            config=config,
            use_chat_template=use_chat_template,
            use_simple_prompt=use_simple_prompt,
            session_info=session_info  # Use trained model's system prompt for fair comparison
        )
        results["base"] = base_result

        # Add comparison metrics
        if trained_result["success"] and base_result["success"]:
            results["comparison"] = {
                "trained_length": len(trained_result["response"]),
                "base_length": len(base_result["response"]),
                "length_diff": len(trained_result["response"]) - len(base_result["response"]),
                "trained_time": trained_result["metadata"]["generation_time"],
                "base_time": base_result["metadata"]["generation_time"],
                "time_diff": trained_result["metadata"]["generation_time"] - base_result["metadata"]["generation_time"]
            }

        return results

    def clear_model_cache(self, model_key: Optional[str] = None):
        """Clear cached models to free memory.

        Args:
            model_key: Specific model to clear, or None for all
        """

        if model_key:
            if model_key in self.loaded_models:
                # Delete the model
                del self.loaded_models[model_key]

                # Force garbage collection
                gc.collect()

                # Clear CUDA cache if available
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                logger.info(f"Cleared model cache for {model_key}")
        else:
            # Clear all models
            self.loaded_models.clear()

            # Force garbage collection
            gc.collect()

            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("Cleared all model cache")

    def set_chat_template(self, template: str):
        """Set custom chat template for base model.

        Args:
            template: Chat template string with {prompt} placeholder
        """
        self.base_chat_template = template
        logger.info("Updated base model chat template")

    def get_loaded_models(self) -> List[Dict[str, str]]:
        """Get list of currently loaded models.

        Returns:
            List of loaded model info
        """
        models = []
        for key, data in self.loaded_models.items():
            model_type, model_id = key.split("_", 1)
            models.append({
                "type": model_type,
                "id": model_id,
                "loaded_at": data["loaded_at"]
            })
        return models
