"""
Router module for SWE Training Framework.

This module provides routing and communication capabilities for distributed model inference
in the software engineering training pipeline. It handles request routing, load balancing,
and asynchronous communication with model servers for generating agent responses.

Key Components:
- Router: Main class for managing model server communication
- Request routing and load balancing across multiple server addresses
- Asynchronous HTTP communication with model inference servers
- Protocol conversion between internal formats and OpenAI-compatible APIs
- Error handling and retry mechanisms for robust distributed inference
- Performance monitoring and logging for debugging and optimization

The router integrates with the veRL protocol system and provides a clean interface
for agent execution engines to communicate with distributed model servers. It supports
both synchronous and asynchronous operations with configurable timeouts and retry logic.

Features:
- Multi-server load balancing and failover
- OpenAI-compatible API integration
- Asynchronous request processing for improved throughput
- Automatic retry mechanisms with exponential backoff
- Comprehensive error handling and logging
- Protocol conversion utilities for seamless integration

Usage:
    router = Router(config=config, tokenizer=tokenizer, addresses=server_addresses)
    response = await router.generate_sequences(batch, application_id="app_123")
    
The router is designed to work seamlessly with the AgentExecutionEngine and provides
the communication layer between agents and distributed model inference servers.
"""

import asyncio
import logging
from copy import deepcopy

import aiohttp
import numpy as np
import torch
from openai.types.completion import Completion
from tensordict import TensorDict

from verl.protocol import DataProto
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length

logger = logging.getLogger(__name__)


def _repeat_interleave(value: torch.Tensor | np.ndarray, repeats: int) -> torch.Tensor | np.ndarray:
    """
    Repeat elements of a tensor or numpy array along the first dimension.
    
    This utility function provides a unified interface for repeating elements
    in both PyTorch tensors and NumPy arrays. Each element in the input is
    repeated 'repeats' number of times along dimension 0.
    
    Args:
        value (torch.Tensor | np.ndarray): Input tensor or array to repeat
        repeats (int): Number of times to repeat each element
        
    Returns:
        torch.Tensor | np.ndarray: Output with same type as input, where each
            element is repeated 'repeats' times along the first dimension
            
    Example:
        >>> import torch
        >>> import numpy as np
        >>> 
        >>> # PyTorch tensor example
        >>> tensor = torch.tensor([[1, 2], [3, 4]])
        >>> result = _repeat_interleave(tensor, 2)
        >>> # Result: [[1, 2], [1, 2], [3, 4], [3, 4]]
        >>> 
        >>> # NumPy array example  
        >>> array = np.array([[1, 2], [3, 4]])
        >>> result = _repeat_interleave(array, 2)
        >>> # Result: [[1, 2], [1, 2], [3, 4], [3, 4]]
        
    Note:
        This function maintains the original data type and device (for tensors)
        of the input while performing the repeat operation.
    """
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    elif isinstance(value, np.ndarray):
        return np.repeat(value, repeats, axis=0)


async def poll_completions_openai(address: str, **completions_request) -> Completion:
    """
    Poll completions from OpenAI-compatible API endpoint asynchronously.
    
    This function sends a completion request to an OpenAI-compatible API server
    and handles retries with exponential backoff in case of failures. It uses
    aiohttp for non-blocking HTTP requests to avoid potential blocking issues
    with the AsyncOpenAI client.
    
    Args:
        address (str): The server address (host:port) to send the request to
        **completions_request: Keyword arguments containing the completion request
            parameters. Common parameters include:
            - prompt (str): The input prompt for completion
            - max_tokens (int): Maximum number of tokens to generate
            - temperature (float): Sampling temperature
            - top_p (float): Nucleus sampling parameter
            - stop (list): Stop sequences
            - application_id (str): Application identifier (removed from request)
            - meta_info (dict): Metadata (removed from request)
            - extra_headers (dict): Additional headers (removed from request)
    
    Returns:
        Completion: OpenAI Completion object containing the generated response
        
    Raises:
        Exception: If all retry attempts fail or if the API returns an error status
        
    Example:
        >>> completion = await poll_completions_openai(
        ...     "localhost:8000",
        ...     prompt="Hello, world!",
        ...     max_tokens=100,
        ...     temperature=0.7
        ... )
        >>> print(completion.choices[0].text)
        
    Note:
        - The function automatically removes 'meta_info', 'extra_headers', and 
          'application_id' from the request payload as they are not part of the
          standard OpenAI API
        - Implements exponential backoff with up to 3 retry attempts
        - Uses a 45-minute timeout (2700 seconds) for long-running requests
        - Creates a new aiohttp session for each request to avoid connection reuse issues
    """
    # Use aiohttp directly instead of AsyncOpenAI to avoid potential blocking
    base_url = f"http://{address}/v1/completions"
    headers = {
        "Content-Type": "application/json",
    }

    # Remove meta_info if present
    if "meta_info" in completions_request:
        completions_request.pop("meta_info")
    # Remove extra_headers from the payload
    if "extra_headers" in completions_request:
        completions_request.pop("extra_headers")
    application_id = "none"
    if "application_id" in completions_request:
        application_id = completions_request.pop("application_id")

    max_retries = 3
    retry_delay = 1  # Initial delay in seconds
    print(f"API request params is : application_id is {application_id}, base_url={base_url}, headers={headers}")
    
    for retry in range(max_retries):
        try:
            # Create a new session for each request to avoid blocking
            async with aiohttp.ClientSession() as session:
                async with session.post(base_url, json=completions_request, headers=headers, timeout=aiohttp.ClientTimeout(total=2700)) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        print(f"API request params is : base_url={base_url}, headers={headers}")
                        raise Exception(f"API request failed with status {response.status}: {error_text}")
                    result = await response.json()
                    # Convert the raw JSON response to an OpenAI Completion object
                    return result
        except Exception as e:
            import traceback

            traceback.print_exc()
            # If this is the last retry, raise the exception
            if retry == max_retries - 1:
                raise e
            # Exponential backoff
            await asyncio.sleep(retry_delay)
            retry_delay *= 2

    # This should never be reached due to the raise in the loop, but mypy requires it
    raise Exception("All retries failed")


class Router:
    """
    Router chooses the least-used server address from a static list of
    server addresses across multiple processes using asyncio locks.
    """

    def __init__(self, config, tokenizer, addresses: list[str]):
        """
        Initialize the Router.
        
        This constructor sets up the routing infrastructure for distributed model
        inference. It initializes the server address list, usage counters, and
        application-to-address mappings for load balancing.
        """
        # List of "ip:port" strings
        self.addresses = addresses
        self.tensor_parallel_size = config.actor_rollout_ref.rollout.get("tensor_model_parallel_size", 1)
        self._lock = asyncio.Lock()
        self._usage: dict[str, int] = {}
        self._application_id_to_address: dict[str, str] = {}
        # Initialize usage counts for any new addresses
        for addr in self.addresses:
            if addr not in self._usage:
                self._usage[addr] = 0
        self.counter = 0
        self.config = config
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id
        model_path = config.actor_rollout_ref.model.path
        self.model_name = "/".join(model_path.split("/")[-2:])

    async def get_address(self, application_id: str) -> str:
        """
        Pick the server address with the smallest usage count and increment its counter.
        """
        async with self._lock:
            min_address, min_usage = min(self._usage.items(), key=lambda x: x[1])
            if application_id not in self._application_id_to_address:
                self._application_id_to_address[application_id] = min_address
                self._usage[min_address] += 1
            else:
                # Data locality
                cur_address = self._application_id_to_address[application_id]
                cur_usage = self._usage[cur_address]
                # Load balance if there is skew
                if (min_usage == 0 or cur_usage - min_usage >= 4) and cur_usage > 0:
                    self._application_id_to_address[application_id] = min_address
                    self._usage[min_address] += 1
                else:
                    self._usage[cur_address] += 1
        return self._application_id_to_address[application_id]

    async def release_address(self, addr: str, application_id: str) -> None:
        """
        Decrement the usage count for a server address when done.
        """
        async with self._lock:
            self._usage[addr] = max(0, self._usage.get(addr, 0) - 1)

    async def generate_sequences(self, batch: DataProto, application_id: str, **sampling_params):
        """
        Generate sequences from the model using the router.
        
        This method processes a batch of data through the router to generate
        sequences. It handles request routing, load balancing, and asynchronous
        communication with model servers. It also manages application-specific
        parameters and metadata for distributed inference.
        """
        kwargs = dict(
            n=self.config.actor_rollout_ref.rollout.n,
            max_tokens=self.config.actor_rollout_ref.rollout.response_length,  # Changed from max_completion_tokens
            temperature=self.config.actor_rollout_ref.rollout.temperature,
            top_p=self.config.actor_rollout_ref.rollout.top_p,
            logprobs=1,
        )

        do_sample = batch.meta_info.get("do_sample", True)
        is_validate = batch.meta_info.get("validate", False)
        if not do_sample or is_validate:
            kwargs["n"] = 1
            kwargs["temperature"] = 0

        if is_validate:
            kwargs.update(
                {
                    #'top_k': self.config.val_kwargs.top_k,
                    "top_p": self.config.actor_rollout_ref.rollout.val_kwargs.top_p,
                    "temperature": self.config.actor_rollout_ref.rollout.val_kwargs.temperature,
                    "n": 1,  # if validate, already repeat in ray_trainer
                }
            )

        if batch.meta_info.get("max_tokens", None) is not None:
            kwargs["max_tokens"] = batch.meta_info["max_tokens"]

        if batch.meta_info.get("agent_rollout", False):
            kwargs["n"] = 1

        kwargs.update(sampling_params)
        
        address = await self.get_address(application_id)
        if "idx" in kwargs:
            idx = kwargs.get("idx", "none")
            del kwargs["idx"]
        else:
            idx = "none"
        kwargs["application_id"] = application_id
        
        tasks = []
        # Bug: len(batch) is used later but batch might not have a __len__ method
        batch_size = len(batch.non_tensor_batch["formatted_prompts"])
        batch_response_ids: list[list[int]] = [[] for _ in range(batch_size)]

        print(f"[TrainingLogsRouter] current idx is {idx}, current application id is {application_id}, request address is {address}, model is {self.model_name}, kwargs is {kwargs}")
        
        import time
        start_time = time.time()
        for batch_index, formatted_prompt in enumerate(batch.non_tensor_batch["formatted_prompts"]):
            # For Completion API, we need to convert the conversation to a prompt string
            self.counter += 1
            tasks.append(
                self.submit_completions(  # Changed from submit_chat_completions
                    address=address,
                    model=self.model_name,
                    prompt=formatted_prompt,  # Changed from messages
                    **kwargs,
                )
            )
        
        

        # Potential blocking: asyncio.gather can block if any task takes too long
        logger.debug("Sending total requests: %s", self.counter)
        completions_list = await asyncio.gather(*tasks)
        await self.release_address(address, application_id)  # Release the address when done

        for batch_index, completions in enumerate(completions_list):
            comps = []
            for choice in completions.get("choices", []):
                token_ids = choice.get("logprobs", {}).get("tokens", [])
                token_ids = [int(t.split(":")[1]) for t in token_ids]
                comps.append(token_ids)
            batch_response_ids[batch_index] = comps
        end_time = time.time()
        print(f"[TrainingLogsRouter] current idx is {idx}, current application id is {application_id}, request address is {address}, model is {self.model_name}, kwargs is {kwargs}, cost time {end_time - start_time}")

        # Extract rollout log probs if calculate_log_probs is enabled
        batch_rollout_log_probs = None
        if self.config.actor_rollout_ref.rollout.calculate_log_probs:
            batch_rollout_log_probs = []
            for batch_index, completions in enumerate(completions_list):
                rollout_log_probs = []
                for choice in completions.get("choices", []):
                    logprobs_data = choice.get("logprobs", {})
                    token_logprobs = logprobs_data.get("token_logprobs", [])
                    tokens = logprobs_data.get("tokens", [])

                    # vLLM returns prompt+response logprobs, slice to get response only
                    if tokens and token_logprobs:
                        response_length = len(tokens)
                        response_logprobs = token_logprobs[-response_length:] if len(token_logprobs) >= response_length else token_logprobs
                        rollout_log_probs.append(response_logprobs)
                    else:
                        rollout_log_probs.append(token_logprobs)
                batch_rollout_log_probs.append(rollout_log_probs)

        return await self.postprocess_batch(batch, batch_response_ids, kwargs["n"], batch_rollout_log_probs)

    async def submit_completions(self, address, model, prompt, **kwargs):
        """
        Submit a completion request to the model server asynchronously.
        
        This method sends a completion request to the specified model server
        and returns the generated response asynchronously. It handles retries
        with exponential backoff in case of failures and uses aiohttp for
        non-blocking HTTP requests.
        """
        # Potential blocking: network I/O can block
        return await poll_completions_openai(address=address, model=model, prompt=prompt, **kwargs)

    async def postprocess_batch(self, batch: DataProto, response_ids: list[list[int]], n: int, rollout_log_probs: list[list[list[float]]] = None) -> DataProto:
        """
        Postprocess the batch of responses from the model server.
        
        This method processes the generated responses from the model server
        and formats them into a DataProto object. It handles padding, repetition,
        and other formatting requirements for distributed inference.
        """
        # NOTE: For Completion API, batch_completions is a list of lists of strings (not dictionaries)
        # prompts: left pad
        # responses: right pad
        # input_ids: prompt + response
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]

        # prompts: [prompt] from input dataset
        idx = batch.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = batch.batch["attention_mask"]
        position_ids = batch.batch["position_ids"]
        non_tensor_batch = deepcopy(batch.non_tensor_batch)

        # Flatten to list.
        # Flatten the list of lists of token IDs
        response = []
        for r_ids in response_ids:
            if r_ids is not None:  # Ensure we don't process None values
                for r in r_ids:
                    response.append(r)
        assert len(response) == len(non_tensor_batch["formatted_prompts"]) * n
        response_tensor = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.actor_rollout_ref.rollout.response_length).to(idx.device)

        if n > 1:
            idx = _repeat_interleave(idx, n)
            attention_mask = _repeat_interleave(attention_mask, n)
            position_ids = _repeat_interleave(position_ids, n)
            for key, val in non_tensor_batch.items():
                non_tensor_batch[key] = _repeat_interleave(val, n)

        batch_size = len(idx)
        seq = torch.cat([idx, response_tensor], dim=-1)

        response_length = response_tensor.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(response_id=response_tensor, eos_token=self.eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # Process rollout log probs if provided
        rollout_log_probs_tensor = None
        if rollout_log_probs is not None:
            # Flatten rollout_log_probs similar to response_ids
            flat_rollout_log_probs = []
            for rollout_log_probs_per_prompt in rollout_log_probs:
                if rollout_log_probs_per_prompt is not None:
                    for rollout_log_prob in rollout_log_probs_per_prompt:
                        flat_rollout_log_probs.append(rollout_log_prob)

            # Pad to same length as response_tensor
            rollout_log_probs_tensor = pad_2d_list_to_length(
                flat_rollout_log_probs,
                0.0,  # Pad with 0.0 for log probs
                max_length=self.config.actor_rollout_ref.rollout.response_length
            ).to(idx.device)

        output = TensorDict(
            {
                "prompts": idx,
                "responses": response_tensor,
                "input_ids": seq,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )

        # Add rollout log probs if available
        if rollout_log_probs_tensor is not None:
            output["rollout_log_probs"] = rollout_log_probs_tensor

        return DataProto(batch=output, meta_info=batch.meta_info)