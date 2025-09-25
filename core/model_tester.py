"""Model testing module for comparing trained models with base models."""

import os
import json
import torch
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from datetime import datetime

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextStreamer,
    GenerationConfig
)
from peft import PeftModel

logger = logging.getLogger(__name__)


@dataclass
class TestConfig:
    """Configuration for model testing."""
    temperature: float = 0.7
    max_new_tokens: int = 512
    top_p: float = 0.95
    top_k: int = 50
    repetition_penalty: float = 1.0
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

            # Check if it's a LoRA model
            adapter_config_path = checkpoint_path / "adapter_config.json"
            if adapter_config_path.exists():
                # Load base model and LoRA adapter
                with open(adapter_config_path, 'r') as f:
                    adapter_config = json.load(f)
                base_model_name = adapter_config.get("base_model_name_or_path")

                logger.info(f"Loading base model: {base_model_name}")
                model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    low_cpu_mem_usage=True
                )

                logger.info("Loading LoRA adapter")
                model = PeftModel.from_pretrained(model, checkpoint_path)
            else:
                # Load full model
                model = AutoModelForCausalLM.from_pretrained(
                    checkpoint_path,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    low_cpu_mem_usage=True
                )

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

            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                low_cpu_mem_usage=True
            )

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

    def generate_response(
        self,
        prompt: str,
        model_type: str,
        model_key: str,
        config: Optional[TestConfig] = None,
        use_chat_template: bool = True,
        streaming_callback = None
    ) -> Dict[str, Any]:
        """Generate response from a model.

        Args:
            prompt: Input prompt
            model_type: "trained" or "base"
            model_key: Session ID for trained or model name for base
            config: Generation configuration
            use_chat_template: Whether to apply chat template
            streaming_callback: Callback for streaming tokens

        Returns:
            Dictionary with response and metadata
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

            # Apply chat template if requested
            if use_chat_template:
                if model_type == "base":
                    # Use generic chat template for base model
                    formatted_prompt = self.base_chat_template.format(prompt=prompt)
                else:
                    # Try to use model's chat template if available
                    if hasattr(tokenizer, 'apply_chat_template'):
                        messages = [{"role": "user", "content": prompt}]
                        formatted_prompt = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                    else:
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

            if self.device == "cuda":
                inputs = inputs.to("cuda")

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

            # Generate response
            start_time = datetime.now()

            with torch.no_grad():
                generation_config = config.to_generation_config()
                outputs = model.generate(
                    **inputs,
                    generation_config=generation_config,
                    streamer=streamer
                )

            end_time = datetime.now()
            generation_time = (end_time - start_time).total_seconds()

            # Decode response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Remove the prompt from response
            if response.startswith(formatted_prompt):
                response = response[len(formatted_prompt):].strip()

            # Calculate token counts
            input_tokens = len(inputs['input_ids'][0])
            output_tokens = len(outputs[0]) - input_tokens

            return {
                "success": True,
                "response": response,
                "metadata": {
                    "model_type": model_type,
                    "model_key": model_key,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                    "generation_time": generation_time,
                    "formatted_prompt": formatted_prompt,
                    "config": {
                        "temperature": config.temperature,
                        "max_new_tokens": config.max_new_tokens,
                        "top_p": config.top_p,
                        "top_k": config.top_k
                    }
                }
            }

        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def compare_models(
        self,
        prompt: str,
        trained_session_id: str,
        base_model_name: str,
        config: Optional[TestConfig] = None,
        use_chat_template: bool = True
    ) -> Dict[str, Any]:
        """Compare responses from trained and base models.

        Args:
            prompt: Input prompt
            trained_session_id: Session ID of trained model
            base_model_name: Name of base model
            config: Generation configuration
            use_chat_template: Whether to apply chat template

        Returns:
            Dictionary with comparison results
        """
        results = {
            "prompt": prompt,
            "timestamp": datetime.now().isoformat(),
            "config": config.__dict__ if config else TestConfig().__dict__
        }

        # Generate from trained model
        trained_result = self.generate_response(
            prompt=prompt,
            model_type="trained",
            model_key=trained_session_id,
            config=config,
            use_chat_template=use_chat_template
        )
        results["trained"] = trained_result

        # Generate from base model
        base_result = self.generate_response(
            prompt=prompt,
            model_type="base",
            model_key=base_model_name,
            config=config,
            use_chat_template=use_chat_template
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
                del self.loaded_models[model_key]
                torch.cuda.empty_cache()
                logger.info(f"Cleared model cache for {model_key}")
        else:
            self.loaded_models.clear()
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