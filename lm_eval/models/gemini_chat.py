"""
Gemini API support via OpenAI compatibility layer.

This module provides custom model classes for Gemini models that remove
unsupported parameters from the OpenAI-compatible API requests.

Includes both completion and chat completion variants.
"""

import logging
import os
from functools import cached_property
from typing import Dict, List, Optional, Union

from lm_eval.api.registry import register_model
from lm_eval.models.openai_completions import LocalChatCompletion, LocalCompletionsAPI

eval_logger = logging.getLogger(__name__)


@register_model("gemini-chat")
class GeminiChatCompletion(LocalChatCompletion):
    """
    Gemini-specific chat completions wrapper.

    This class extends LocalChatCompletion to remove parameters that are
    not supported by Gemini's OpenAI compatibility layer, such as 'seed'.

    Supports Gemini thinking models (e.g., gemini-2.0-flash-thinking-exp).
    When using thinking models, the thinking process will be included in the response.

    Usage:
        lm_eval --model gemini-chat \
            --model_args model=gemini-2.5-flash,base_url=https://generativelanguage.googleapis.com/v1beta/openai/chat/completions,tokenizer_backend=huggingface,tokenizer=Xenova/gpt-4 \
            --tasks hellaswag \
            --batch_size 1

        # For thinking models with reasoning effort:
        lm_eval --model gemini-chat \
            --model_args model=gemini-2.0-flash-thinking-exp,base_url=https://generativelanguage.googleapis.com/v1beta/openai/chat/completions,tokenizer_backend=huggingface,tokenizer=Xenova/gpt-4,reasoning_effort=high \
            --tasks hellaswag \
            --batch_size 1
    """

    def __init__(
        self,
        reasoning_effort: str = None,
        **kwargs,
    ):
        """
        Initialize Gemini chat completion model.

        Args:
            reasoning_effort: Controls thinking effort for thinking models.
                             Valid values: "low", "medium", "high"
                             Only applicable to thinking models like gemini-2.0-flash-thinking-exp
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(**kwargs)
        self.reasoning_effort = reasoning_effort

        if reasoning_effort:
            valid_efforts = ["low", "medium", "high"]
            if reasoning_effort not in valid_efforts:
                eval_logger.warning(
                    f"reasoning_effort='{reasoning_effort}' is not in {valid_efforts}. "
                    "This may cause API errors if the model doesn't support this value."
                )

    def _create_payload(
        self,
        messages: List[Dict],
        generate=False,
        gen_kwargs: dict = None,
        seed=1234,
        eos=None,
        **kwargs,
    ) -> dict:
        """
        Create API payload for Gemini, excluding unsupported parameters.

        Gemini's OpenAI compatibility layer doesn't support:
        - seed parameter
        - Some other OpenAI-specific parameters
        """
        # Call parent method to get the standard payload
        payload = super()._create_payload(
            messages=messages,
            generate=generate,
            gen_kwargs=gen_kwargs,
            seed=seed,
            eos=eos,
            **kwargs,
        )

        # Remove unsupported parameters for Gemini
        payload.pop("seed", None)

        # Add reasoning_effort for thinking models if specified
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort

        # Log once if we haven't already
        if not hasattr(self, "_logged_gemini_warning"):
            log_msg = "Using Gemini-specific API adapter. Removed 'seed' parameter from requests."
            if self.reasoning_effort:
                log_msg += f" Added reasoning_effort='{self.reasoning_effort}'."
            eval_logger.info(log_msg)
            self._logged_gemini_warning = True

        return payload


@register_model("gemini-completions")
class GeminiCompletionsAPI(LocalCompletionsAPI):
    """
    Gemini-specific completions wrapper for prompt completion endpoint.

    This class extends LocalCompletionsAPI to support loglikelihood-based tasks
    (multiple choice, perplexity) using Gemini's /completions endpoint.

    Key differences from gemini-chat:
    - Supports loglikelihood tasks (MMLU, HellaSwag, etc.)
    - Uses /completions endpoint instead of /chat/completions
    - Input format: prompt (string) instead of messages (list of dicts)
    - Does NOT support thinking models or reasoning_effort

    When to use:
    - Use gemini-completions: For tasks requiring loglikelihood (multiple choice, perplexity)
    - Use gemini-chat: For generative tasks, chat templates, thinking models

    Usage:
        lm_eval --model gemini-completions \
            --model_args model=gemini-2.5-flash,base_url=https://generativelanguage.googleapis.com/v1beta/openai/completions,tokenizer_backend=huggingface,tokenizer=Xenova/gpt-4 \
            --tasks hellaswag \
            --batch_size 1
    """

    def __init__(
        self,
        base_url: str = None,
        tokenizer_backend: str = "huggingface",
        **kwargs,
    ):
        """
        Initialize Gemini completions model.

        Args:
            base_url: API endpoint URL. Defaults to Gemini's completions endpoint.
            tokenizer_backend: Tokenizer to use. Defaults to 'huggingface'.
                              Use 'huggingface' with tokenizer='Xenova/gpt-4'.
                              Do NOT use 'tiktoken' (doesn't recognize Gemini models)
                              Do NOT use 'remote' (Gemini doesn't expose /tokenizer_info)
            **kwargs: Additional arguments passed to parent class
        """
        # Set default base_url if not provided
        if base_url is None:
            base_url = "https://generativelanguage.googleapis.com/v1beta/openai/completions"

        super().__init__(
            base_url=base_url,
            tokenizer_backend=tokenizer_backend,
            **kwargs,
        )

    def _create_payload(
        self,
        messages: Union[List[List[int]], List[dict], List[str], str],
        generate: bool = False,
        gen_kwargs: Optional[dict] = None,
        seed: int = 1234,
        eos: str = None,
        **kwargs,
    ) -> dict:
        """
        Create API payload for Gemini completions, excluding unsupported parameters.

        Gemini's OpenAI compatibility layer doesn't support:
        - seed parameter
        - Some other OpenAI-specific parameters

        Args:
            messages: Prompt text or tokens
            generate: Whether this is a generation request
            gen_kwargs: Generation parameters
            seed: Random seed (will be removed for Gemini)
            eos: End of sequence token
            **kwargs: Additional parameters

        Returns:
            dict: API request payload
        """
        # Call parent method to get the standard payload
        payload = super()._create_payload(
            messages=messages,
            generate=generate,
            gen_kwargs=gen_kwargs,
            seed=seed,
            eos=eos,
            **kwargs,
        )

        # Remove unsupported parameters for Gemini
        payload.pop("seed", None)

        # Log once if we haven't already
        if not hasattr(self, "_logged_gemini_warning"):
            eval_logger.info(
                "Using Gemini completions API. Removed 'seed' parameter from requests."
            )
            self._logged_gemini_warning = True

        return payload

    @cached_property
    def api_key(self) -> str:
        """
        Get API key from environment variable.

        Returns:
            str: API key

        Raises:
            ValueError: If OPENAI_API_KEY environment variable is not set
        """
        key = os.environ.get("OPENAI_API_KEY", None)
        if key is None:
            raise ValueError(
                "API key not found. Please set the OPENAI_API_KEY environment variable "
                "to your Gemini API key from https://aistudio.google.com/apikey"
            )
        return key
