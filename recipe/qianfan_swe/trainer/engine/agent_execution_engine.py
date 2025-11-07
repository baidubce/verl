"""
Agent Execution Engine for SWE Training Framework.

This module provides the core execution engine for running software engineering agents
in a distributed training environment. It handles agent lifecycle management, tool
execution, trajectory generation, and result collection for PPO training.

Key Components:
- AgentExecutionEngine: Main engine for executing agent rollouts
- Container management for isolated execution environments
- Tool integration and execution tracking
- Trajectory collection and formatting for training
- Performance monitoring and logging

The engine supports both Docker and Kubernetes execution environments and provides
comprehensive error handling and timeout management for robust training workflows.
"""

import asyncio
import concurrent.futures
import logging
import time
import traceback
import uuid
import json
from concurrent.futures import ThreadPoolExecutor

import sys
import os
import numpy as np
import torch
from contextlib import contextmanager

from recipe.qianfan_swe.trainer.utils.utils import (
    convert_messages_to_tokens_and_masks
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recipe.qianfan_swe.trainer.utils.misc import colorful_print
from recipe.qianfan_swe.trainer.router.router import Router
from recipe.qianfan_swe.trainer.utils.parser import ChatTemplateParser

logger = logging.getLogger(__name__)

from recipe.qianfan_swe.trainer.utils.agent_utils import _create_agent

@contextmanager
def log_time(desc="耗时"):
    """
    时间记录上下文管理器，用于性能监控和调试。
    
    Args:
        desc (str): 描述信息，默认为"耗时"
        
    Usage:
        with log_time("数据库查询"):
            # 执行需要计时的代码
            pass
    
    输出格式: [DockerTimeLog] {desc}: {elapsed:.2f}秒
    """
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"[DockerTimeLog] {desc}: {elapsed:.2f}秒")
    
    
def summarize_wandb_result(wandb_result, is_training=True):
    """
    统计wandb_result中每个key的均值、总和、最大值、最小值，输出为json格式。
    每个统计项的key格式为"rollout_info/{原key}_{stat}"，如"rollout_info/rollout_cost_time_mean"
    """
    if not wandb_result:
        return {}

    # 收集所有key
    keys = set()
    for item in wandb_result:
        keys.update(item.keys())

    stats = {}
    for k in keys:
        values = [item[k] for item in wandb_result if k in item and isinstance(item[k], (int, float))]
        if not values:
            continue
        if is_training:
            stats[f"rollout_info/{k}_mean"] = sum(values) / len(values)
            stats[f"rollout_info/{k}_sum"] = sum(values)
            stats[f"rollout_info/{k}_max"] = max(values)
            stats[f"rollout_info/{k}_min"] = min(values)
        else:
            stats[f"rollout_info_val/{k}_mean"] = sum(values) / len(values)
            stats[f"rollout_info_val/{k}_sum"] = sum(values)
            stats[f"rollout_info_val/{k}_max"] = max(values)
            stats[f"rollout_info_val/{k}_min"] = min(values)

    return stats



class AgentExecutionEngine:
    def __init__(
        self,
        engine_name="verl",
        tokenizer=None,
        rollout_engine=None,
        chat_parser=None,
        n_parallel_agents=1,
        trajectory_timeout=None,
        gamma=0.2,
        api_retries=5,
        retry_limit=10,
        max_steps=5,
        max_response_length=8192,
        max_prompt_length=1024,
        config=None,
        rollout_engine_args=None,
        env_args=None,
        max_workers=256,
        enforce_max_prompt_length=False,  # If enabled, applies max_prompt check per step
        overlong_filter=False,  # Filter for overlong trajectories (i.e. TRUNCATION, MAX_STEPS, TIMEOUT)
        logger=None,
        extra_infos=None,
        raw_prompts=None,
        pod_managers=None,  # List of PodManager instances for managing Kubernetes pods
        **kwargs,
    ):
        if rollout_engine_args is None:
            rollout_engine_args = {}
        if env_args is None:
            env_args = {}
        
        self.config = config
        self.rollout_engine = rollout_engine
        self.tokenizer = tokenizer
        self.engine_name = engine_name
        self.n_parallel_agents = n_parallel_agents
        self.overlong_filter = overlong_filter
        self.logger = logger
        self.extra_infos = extra_infos
        self.raw_prompts = raw_prompts
        self.pod_managers = pod_managers or []
        # For interaction
        self.gamma = gamma
        self.retry_limit = retry_limit
        self.api_retries = api_retries
        self.max_steps = max_steps
        self.max_response_length = max_response_length
        self.max_prompt_length = max_prompt_length
        self.enforce_max_prompt_length = enforce_max_prompt_length

        self.env_args = env_args

        # Workers agent configuration

        self.agents = [None for _ in range(n_parallel_agents)]
        self.envs = [None for _ in range(n_parallel_agents)]
        

        self.trajectory_timeout = trajectory_timeout
        if not trajectory_timeout:
            self.trajectory_timeout = int(1e9)
        # rollout engine args
        self.rollout_engine_args = rollout_engine_args
        self.sampling_params = kwargs.get("sampling_params", {})

        assert self.engine_name in ["verl"], "Currently only verl are supported as rollout engine"
        if self.engine_name == "verl":
            # All generation is done via scheduler. Currently only works for verl
            self.server_addresses = getattr(self.rollout_engine, "server_addresses", [])
            print("self.server_addresses is : ", self.server_addresses)
            self.router = Router(config=self.config, tokenizer=self.tokenizer, addresses=self.server_addresses)

        # Create a thread pool executor for environment interactions (i.e. step, reset, close)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        
        # Initialize chat parser - assuming it's available from the utils
        if chat_parser is None:
            # Try to import and initialize chat parser
            try:
                self.chat_parser = ChatTemplateParser.get_parser(self.tokenizer, disable_thinking=kwargs.get("disable_thinking", False))
            except ImportError:
                # Fallback to a simple parser if not available
                self.chat_parser = None
                print("Warning: ChatTemplateParser not available, using fallback")
        else:
            self.chat_parser = chat_parser

        # Cache for storing inference log probs per application_id for IcePop
        self.rollout_log_probs_cache = {}


    async def get_model_response(self, messages, application_id, **kwargs):
        """
        Compute model response asynchronously based on the engine type.

        This function is multithread safe and routes the request to the appropriate
        engine-specific handler.

        Args:
            prompt: The input prompt to send to the model
            application_id: Unique identifier for the application
            **kwargs: Additional arguments to pass to the model

        Returns:
            str: The model's response text (log probs are cached for per-step alignment)

        Raises:
            NotImplementedError: If the engine type is not supported
        """
        if self.engine_name == "verl":
            # _get_verl_async returns response string (log probs are cached internally)
            response = await self._get_verl_async(messages, application_id, **kwargs)
            return response
        else:
            raise NotImplementedError(f"Engine type '{self.engine_name}' not supported")

    def update_envs_and_agents(self, agents, extra_infos, pod_names, images, raw_prompts, pod_managers):
        """
        Update the environments and agents.

        Args:
            agents: List of agents to use
            pod_managers: List of PodManager instances, one for each agent
        """
        # For keeping track of the environment index in the batch.
        self.agents = agents
        self.n_parallel_agents = len(agents)
        self.extra_infos = extra_infos
        self.pod_names = pod_names
        self.images = images
        self.raw_prompts = raw_prompts
        self.pod_managers = pod_managers

    async def _get_verl_async(self, messages, application_id, **kwargs):
        """
        Get model response using veRL engine asynchronously.
        
        This method converts the input messages to veRL DataProto format,
        sends them to the model router for generation, and processes the
        response to extract clean text output.
        
        Args:
            messages: Input messages in chat format to send to the model
            application_id: Unique identifier for the application request
            **kwargs: Additional arguments including:
                - max_tokens: Maximum number of tokens to generate
                - Other generation parameters passed to the router
        
        Returns:
            str: The model's response text with padding and special tokens removed
        
        Raises:
            Exception: If there's an error during generation or processing
        """
        batch = self._convert_prompt_verl([messages], **kwargs)

        if "max_tokens" in kwargs:
            batch.meta_info["max_tokens"] = kwargs["max_tokens"]

        output = await self.router.generate_sequences(batch, application_id=application_id, **kwargs)

        # Extract inference log probs for IcePop if available
        rollout_log_probs = None
        if "rollout_log_probs" in output.batch:
            rollout_log_probs = output.batch["rollout_log_probs"][0]

        attn = output.batch["attention_mask"][0, self.max_prompt_length :]
        tokens = output.batch["responses"][0]

        # Find last index where attention == 1
        non_pad_indices = (attn == 1).nonzero(as_tuple=True)[0]
        if len(non_pad_indices) == 0:
            trimmed = tokens[:0]  # empty
            trimmed_log_probs = None
        else:
            last_valid_idx = non_pad_indices[-1].item()
            trimmed = tokens[: last_valid_idx + 1]  # include the last valid token
            # Trim log probs to match trimmed tokens
            if rollout_log_probs is not None:
                trimmed_log_probs = rollout_log_probs[: last_valid_idx + 1]
            else:
                trimmed_log_probs = None

        response = self.tokenizer.decode(trimmed, skip_special_tokens=False)

        pad_token = self.tokenizer.pad_token
        eos_token = self.tokenizer.eos_token
        response = response.replace(pad_token, "").replace(eos_token, "")

        # Cache log probs for per-step alignment
        if trimmed_log_probs is not None:
            if application_id not in self.rollout_log_probs_cache:
                self.rollout_log_probs_cache[application_id] = []
            self.rollout_log_probs_cache[application_id].append(trimmed_log_probs)

        return response

    async def run_agent_trajectory_async(self, idx, application_id, seed=0, mode="Text", **kwargs):
        """Run a single agent's trajectory asynchronously"""
        pod_name = None
        try:
            agent = self.agents[idx]
            pod_name = self.pod_names[idx]

            # Get the pod_manager for this specific agent
            pod_manager = self.pod_managers[idx] if hasattr(self, 'pod_managers') and idx < len(self.pod_managers) else None

            termination_reason = None
            prompt_token_len = 0
            prompt_tokens = []
            response_token_len = 0
            response_tokens = []
            response_masks = []
            total_time = 0.0
            llm_time = 0.0
            env_time = 0.0
            reward = 0.0
            masked_out = False

            # for step return
            episode_steps = []

            prompt = self.raw_prompts[idx]
            # Extract the first user message from prompt
            if isinstance(prompt, list) and len(prompt) > 0:
                for msg in prompt:
                    if msg.get("role") == "user":
                        prompt = msg.get("content", "")
                        break
            elif isinstance(prompt, dict):
                prompt = prompt.get("content", "")
            elif not isinstance(prompt, str):
                prompt = str(prompt)
            # Reset environment with the task using the executor
            loop = asyncio.get_event_loop()

            # Start timing for the entire trajectory
            trajectory_start_time = time.time()
            
            try:
                trajectory_result = await asyncio.wait_for(
                    agent.run_trajectory(
                        prompt=prompt,
                        llm_generate_func=self._get_verl_async,
                        request_id=f"{pod_name}",
                        tokenizer_func=self.tokenizer,
                        max_tokens=self.max_response_length,
                        chat_parser=self.chat_parser,
                        idx=idx,
                        application_id=application_id,
                        trajectory_timeout=self.trajectory_timeout,
                        convert_messages_to_tokens_and_masks=convert_messages_to_tokens_and_masks,
                        pod_name=pod_name,
                        executor=self.executor,
                        **kwargs
                    ),
                    timeout=(self.trajectory_timeout*2)
                )
                total_time = time.time() - trajectory_start_time
                
                termination_reason = trajectory_result.metadata["stop_reason"]
            except asyncio.TimeoutError:
                total_time = time.time() - trajectory_start_time
                
                # Create a failed result with empty steps
                if agent.trajectory:
                    trajectory_result = agent.trajectory
                    print(f"[TrajectoryLogs] Trajectory timeout after {self.trajectory_timeout} seconds for pod {pod_name}, idx is {idx}, application_id is {application_id}, pod_name is {pod_name}")
                else:
                    raise Exception(f"Current idx is {idx}, No execute run trajectory task!")
                termination_reason = "TIMEOUT"
                reward = 0.0
            
            # Check if agent reached max rounds without calling termination tool
            reached_max_rounds = len(trajectory_result.steps) >= self.max_steps * 2  # Rough estimate: 50 rounds * 2 steps per round
            called_submit = any(
                step.tool_name in agent.termination_tool_names
                for step in trajectory_result.steps 
                if hasattr(step, 'tool_name')
            )
            
            # Determine additional termination reasons
            if termination_reason is None:
                if reached_max_rounds and not called_submit:
                    termination_reason = "MAX_STEPS"
                elif len(trajectory_result.steps) > 0:
                    termination_reason = "ENV_DONE"
                else:
                    termination_reason = "UNKNOWN"
            
            result_messages = agent.format_messages_for_llm(trajectory_result)
            if agent.show_steps_remaining:
                result_messages = agent._add_steps_remaining(result_messages, 0)
            print(f"[TrainingLogs] current result_messages is {result_messages}")

            # 获取第一个user消息以及之前的所有内容
            init_messages = []
            for mes in result_messages:  
                init_messages.append(mes)
                if mes["role"] == "user":
                    break

            # result_messages取剩下的内容
            response_messages = result_messages[len(init_messages): ]
            if not response_messages:
                raise Exception(f"[TrajectoryLogs] No user message found in result_messages, pod {pod_name}, idx is {idx}, application_id is {application_id}, pod_name is {pod_name}")
            
            if self.chat_parser:
                prompt_tokens, _ = convert_messages_to_tokens_and_masks(init_messages, tokenizer=self.tokenizer, parser=self.chat_parser, contains_first_msg=True, contains_generation_msg=True)
            else:
                raise Exception("Please provide chat_parser function !")
            prompt_token_len = len(prompt_tokens)
            
            # Check for prompt truncation
            if prompt_token_len > self.max_prompt_length:
                termination_reason = "PROMPT_TRUNCATION"
                reward = 0.0
                print(f"[TrajectoryLogs] Prompt length {prompt_token_len} exceeds max_prompt_length {self.max_prompt_length}")

            response_tokens = []
            response_masks = []
            response_token_types = []  # Track which tokens are assistant (1) vs environment (0)

            # IcePop: Collect aligned inference log probs per-step
            aligned_rollout_log_probs_list = []

            # Get cached log probs from vLLM (one per assistant message)
            cached_log_probs = []
            if application_id in self.rollout_log_probs_cache:
                cached_log_probs = self.rollout_log_probs_cache[application_id]

            assistant_msg_idx = 0

            for msg in response_messages:
                if self.chat_parser:
                    if msg["role"] == "assistant":
                        response_token, response_mask = convert_messages_to_tokens_and_masks([msg], tokenizer=self.tokenizer, parser=self.chat_parser, contains_first_msg=False, contains_generation_msg=False)
                        response_token_types.extend([1] * len(response_token))  # Mark as assistant

                        # IcePop: Align THIS step's vLLM log probs with re-tokenized assistant tokens
                        if assistant_msg_idx < len(cached_log_probs) and cached_log_probs[assistant_msg_idx] is not None:
                            step_rollout_log_probs = cached_log_probs[assistant_msg_idx]

                            if len(response_token) != step_rollout_log_probs.shape[0]:
                                if step_rollout_log_probs.shape[0] > len(response_token):
                                    aligned_logprobs = step_rollout_log_probs[:len(response_token)]
                                else:
                                    padding_size = len(response_token) - step_rollout_log_probs.shape[0]
                                    aligned_logprobs = torch.nn.functional.pad(
                                        step_rollout_log_probs, (0, padding_size), value=0.0
                                    )
                                aligned_rollout_log_probs_list.append(aligned_logprobs)
                            else:
                                aligned_rollout_log_probs_list.append(step_rollout_log_probs)
                        else:
                            aligned_rollout_log_probs_list.append(None)

                        assistant_msg_idx += 1
                    else:
                        response_token, response_mask = convert_messages_to_tokens_and_masks([msg], tokenizer=self.tokenizer, parser=self.chat_parser, contains_first_msg=False, contains_generation_msg=True)
                        response_token_types.extend([0] * len(response_token))  # Mark as environment
                    response_tokens.extend(response_token)
                    response_masks.extend(response_mask)
                else:
                    raise Exception(f"[TrajectoryLogs] No chat parser found, cannot tokenize response")
                    
            response_token_len = len(response_tokens)
            
            # Check for response truncation
            if response_token_len > self.max_response_length:
                termination_reason = "TRUNCATION"
                original_len = response_token_len
                # Truncate tokens and masks
                response_tokens = response_tokens[:self.max_response_length]
                response_masks = response_masks[:self.max_response_length]
                response_token_types = response_token_types[:self.max_response_length]
                response_token_len = len(response_tokens)
                print(f"[TrajectoryLogs] Response length truncated from {original_len} to {response_token_len}")

            masked_out = False
            if self.overlong_filter:
                if termination_reason in ["TRUNCATION", "MAX_STEPS", "TIMEOUT", "PROMPT_TRUNCATION"]:
                    # Mask out the entire response for overlong trajectories if the reward is 0.
                    response_masks = [0] * len(response_masks)
                    masked_out = True
                    print(f"[TrajectoryLogs] Trajectory masked out due to overlong filter. Reason: {termination_reason}")

            # Calculate reward using PodManager (only if not already set to 0 due to truncation/timeout)
            if not masked_out:
                reward = await loop.run_in_executor(self.executor,
                                                    self._calculate_reward_with_pod_manager, 
                                                    pod_manager,
                                                    pod_name, 
                                                    self.extra_infos[idx]
                )
            
            # Log termination information
            if termination_reason:
                if reward > 0:
                    color_msg = "SUCCESS"
                else:
                    color_msg = "FAILED"
                print(f"[TrajectoryLogs] Trajectory completed with {termination_reason}. Reward: {reward} ({color_msg})")
                if masked_out:
                    print(f"[TrajectoryLogs] Trajectory is masked out due to overlong filter.")

            # Extract and process ALIGNED inference log probs for IcePop
            traj_rollout_log_probs = None
            if aligned_rollout_log_probs_list and any(x is not None for x in aligned_rollout_log_probs_list):
                valid_logprobs = [lp for lp in aligned_rollout_log_probs_list if lp is not None]
                if valid_logprobs:
                    vllm_rollout_log_probs = torch.cat(valid_logprobs, dim=0)
                    padded_rollout_log_probs = torch.zeros(len(response_tokens), dtype=torch.float32)
                    assistant_positions = [i for i, t in enumerate(response_token_types) if t == 1]

                    num_to_fill = min(len(assistant_positions), vllm_rollout_log_probs.shape[0])
                    for i in range(num_to_fill):
                        padded_rollout_log_probs[assistant_positions[i]] = vllm_rollout_log_probs[i]

                    traj_rollout_log_probs = padded_rollout_log_probs

            # Clean up cache (early cleanup for memory efficiency)
            if application_id in self.rollout_log_probs_cache:
                del self.rollout_log_probs_cache[application_id]

            token_result = {
                "prompt_tokens": torch.tensor(prompt_tokens, dtype=torch.long),
                "response_tokens": torch.tensor(response_tokens, dtype=torch.long),
                "response_masks": torch.tensor(response_masks, dtype=torch.long),
                "trajectory_reward": reward,
                "docker": self.images[idx],
                "idx": idx,
                "chat_completions": result_messages,
                "termination_reason": termination_reason,
                "masked_out": float(masked_out),
                "rollout_log_probs": traj_rollout_log_probs,
                "metrics": {
                    # Total number of steps taken in the trajectory
                    "steps": len(trajectory_result.steps)//2 if hasattr(trajectory_result, 'steps') else 0,
                    # Total time spent in the trajectory
                    "total_time": total_time,
                    # Token length information
                    "prompt_token_len": prompt_token_len,
                    "response_token_len": response_token_len,
                },
            }
            return token_result

        except Exception as e:
            stack_trace = traceback.format_exc()
            print(f"[TrainingLogs] func run_agent_trajectory_async, generate trajectory error, error msg is {e}, response params is idx: {idx}, application_id: {application_id}, mode: {mode}, kwargs: {kwargs}, corresponding traceback code is: {stack_trace}")
            raise Exception(f"error msg is {e}, corresponding traceback code is : {stack_trace}")
            
        finally:
            # Ensure cache cleanup even on exceptions (safety net for memory leak prevention)
            if application_id in self.rollout_log_probs_cache:
                del self.rollout_log_probs_cache[application_id]

            # Always kill pod when trajectory completes or fails
            pod_manager = self.pod_managers[idx] if hasattr(self, 'pod_managers') and idx < len(self.pod_managers) else None
            if pod_name and pod_manager:
                try:
                    pod_manager.kill_pod(pod_name)
                    print(f"[PodManager] Cleaned up pod {pod_name} after trajectory completion")
                except Exception as e:
                    print(f"[PodManager] Warning: Failed to cleanup pod {pod_name}: {e}")

    async def run_agent_trajectory_with_retry(self, idx, application_id, seed=0, mode="Text", **kwargs):
        """
        Run agent trajectory with retry mechanism.
        
        This method attempts to execute an agent trajectory with automatic retry
        on failure. It includes pod cleanup and reset functionality for robust
        execution in distributed environments.
        
        Args:
            idx (int): Index of the agent to run
            application_id (str): Unique identifier for the application request
            seed (int, optional): Random seed for reproducibility. Defaults to 0.
            mode (str, optional): Execution mode. Defaults to "Text".
            **kwargs: Additional arguments passed to the trajectory execution
            
        Returns:
            dict: Token result containing trajectory data, rewards, and metrics
            
        Raises:
            Exception: If all retry attempts fail, raises exception with details
            
        Note:
            - Uses asyncio.wait_for with 7200 second timeout per attempt
            - Automatically resets pods between retry attempts
            - Cleans up resources on both success and failure
        """
        for ind in range(self.retry_limit):
            try:
                # Clear stale cache before retry to prevent cache pollution (defense-in-depth)
                if ind > 0 and application_id in self.rollout_log_probs_cache:
                    del self.rollout_log_probs_cache[application_id]

                return await asyncio.wait_for(self.run_agent_trajectory_async(idx, application_id=application_id, seed=seed, mode=mode, **kwargs), timeout=7200)
            except Exception as e:
                stack_trace = traceback.format_exc()
                print(f"[TrainingLogs] func run_agent_trajectory_with_retry, generate trajectory error, error msg is {e}, response params is idx: {idx}, application_id: {application_id}, mode: {mode}, kwargs: {kwargs}, corresponding traceback code is : {stack_trace}, retry is {ind}/{self.retry_limit}")
                traceback.print_exc()

                # Reset pod for retry if PodManager is available and this is not the last attempt
                pod_manager = self.pod_managers[idx] if hasattr(self, 'pod_managers') and idx < len(self.pod_managers) else None
                if pod_manager and ind < self.retry_limit - 1:
                    await self._reset_pod_for_retry(idx, pod_manager)

                continue
        traceback.print_exc()
        raise Exception(f"Trajectory {idx} cannot complete. Please check the log message")

    async def _reset_pod_for_retry(self, idx, pod_manager):
        """
        Reset the pod and agent for a specific agent index during retry.
        This creates a fresh pod environment and agent for the retry attempt.
        
        Args:
            idx: Index of the agent/pod to reset
            pod_manager: PodManager instance for this specific agent (will be replaced)
        """
        try:
            old_pod_name = self.pod_names[idx] if idx < len(self.pod_names) else None
            env_args = [self.extra_infos[idx]] if idx < len(self.extra_infos) else [{}]
            
            if not old_pod_name:
                print(f"[PodManager] Warning: No pod name found for index {idx}, cannot reset")
                return
            
            print(f"[PodManager] Resetting pod {old_pod_name} and agent for retry (index {idx})")
            
            # Kill the old pod first using the existing pod_manager
            try:
                pod_manager.kill_pod(old_pod_name)
                print(f"[PodManager] Killed old pod {old_pod_name}")
            except Exception as e:
                print(f"[PodManager] Warning: Failed to kill old pod {old_pod_name}: {e}")
            
            # Create new agent using _create_agent in executor
            loop = asyncio.get_event_loop()
            new_idx, new_pod_name, new_image, new_agent, new_pod_manager = await loop.run_in_executor(
                self.executor,
                _create_agent,
                idx,
                env_args,
                self.config
            )
            
            # Update all related data structures for this index
            self.agents[idx] = new_agent
            self.pod_names[idx] = new_pod_name  
            self.images[idx] = new_image
            self.pod_managers[idx] = new_pod_manager
            
            print(f"[PodManager] Agent and pod reset completed for index {idx}: {old_pod_name} -> {new_pod_name}")
            
        except Exception as e:
            print(f"[PodManager] Error resetting agent and pod for index {idx}: {e}")
            # Don't raise the exception as this would prevent retry attempts
            # The retry will continue with the existing pod and agent

    async def trajectory_generator(self, reset_seed=0, timing_raw=None, mode="Text", global_steps=0, is_training=True, 
                                   **kwargs):
        """
        Generate trajectories for all agents in parallel.
        
        This is the main trajectory generation method that orchestrates the execution
        of multiple agent trajectories concurrently. It manages the lifecycle of
        trajectory tasks, handles completion tracking, and aggregates metrics.
        
        Args:
            reset_seed (int, optional): Random seed for environment reset. Defaults to 0.
            timing_raw (dict, optional): Dictionary to store timing information. Defaults to None.
            mode (str, optional): Execution mode for the trajectory. Defaults to "Text".
            global_steps (int, optional): Current global training step for logging. Defaults to 0.
            is_training (bool, optional): Whether this is a training run (affects metric naming). Defaults to True.
            **kwargs: Additional keyword arguments passed to trajectory execution.
            
        Yields:
            dict: Individual trajectory results as they complete, containing:
                - trajectory data (observations, actions, rewards, etc.)
                - metadata (timing, success metrics, etc.)
                - agent-specific information
                
        Side Effects:
            - Wakes up and sleeps the rollout engine (for verl engine)
            - Logs aggregated metrics to wandb/logger
            - Manages thread pool executor lifecycle
            - Prints progress updates and training logs
            
        Note:
            This method uses asyncio.as_completed() to yield results as soon as they're
            available, allowing for streaming processing of completed trajectories.
        """
        if timing_raw is None:
            timing_raw = {}
        max_concurrency = self.n_parallel_agents
        self.executor = ThreadPoolExecutor(max_workers=max_concurrency)

        if self.engine_name == "verl":
            self.rollout_engine.wake_up()

        async def launch_one_trajectory_task(env_idx: int):
            try:
                application_id = str(uuid.uuid4())
                result = await self.run_agent_trajectory_with_retry(
                    idx=env_idx,
                    application_id=application_id,
                    seed=reset_seed,
                    mode=mode,
                    **kwargs,
                )
            except Exception as e:
                import traceback

                traceback.print_exc()
                raise e
            return result

        # Create all N conceptual tasks. Their execution will be throttled by the semaphore
        # and the availability of agent/env indices.
        tasks_to_run = [launch_one_trajectory_task(i) for i in range(len(self.agents))]
        wandb_info = []

        tasks_completed = 0
        for coro in asyncio.as_completed(tasks_to_run):
            try:
                result = await coro
                tasks_completed += 1
                colorful_print(f"Number of Trajectories {tasks_completed}/{len(self.agents)} completed", "cyan")
                yield result
            except Exception as e:
                raise e
                
        wandb_info_metric = summarize_wandb_result(wandb_info, is_training=is_training)
        self.logger.log(data=wandb_info_metric, step=global_steps)
        print(f"[TrainingLogs] current training step is {global_steps}, rollout metrics is {wandb_info_metric}")

        if self.engine_name == "verl":
            self.rollout_engine.sleep()

        self.executor.shutdown(wait=False, cancel_futures=True)

    def _convert_prompt_verl(self, prompts, **kwargs):
        """
        Given a list of prompts in Chat template, convert to DataProto format in veRL

        Args:
            prompts: List of prompts to convert
            **kwargs: Additional arguments

        Returns:
            DataProto object containing the converted prompts
        """
        from verl.protocol import DataProto, union_two_dict
        from verl.utils.model import compute_position_id_with_mask
        from verl.utils.torch_functional import pad_sequence_to_length

        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        # Convert messages to formatted prompts
        formatted_prompts = [self.chat_parser.parse(prompt, add_generation_prompt=True, is_first_msg=True) for prompt in prompts]

        # Tokenize the final processed strings
        inputs = self.tokenizer(
            formatted_prompts,
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        self.tokenizer.padding_side = old_padding_side

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # pad to max sizes
        input_ids = pad_sequence_to_length(input_ids, max_seq_len=self.max_prompt_length, pad_token_id=self.tokenizer.pad_token_id, left_pad=True)
        attention_mask = pad_sequence_to_length(attention_mask, max_seq_len=self.max_prompt_length, pad_token_id=0, left_pad=True)
        position_ids = compute_position_id_with_mask(attention_mask)
        batch_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        data = DataProto.from_dict(batch_dict)
        data.non_tensor_batch["formatted_prompts"] = np.array(formatted_prompts)

        # original_batch contains the extra info needed for generation
        if "meta_info" in kwargs and kwargs["meta_info"]:
            meta_info = kwargs["meta_info"]
            # only use the original_batch's meta_info since tensor_batch is from batch_dict and non_tensor_batch is not neeeded
            data.meta_info = union_two_dict(data.meta_info, meta_info)

        return data


    def _calculate_reward_with_pod_manager(self, pod_manager, pod_name, ds):
        """
        Calculate reward using PodManager based on environment type
        
        Args:
            pod_manager: PodManager instance for this specific agent
            pod_name: Name of the pod to execute commands in
            ds: Dataset instance containing dataset and configuration
            alt_path: Alternative path for scripts (default: "/")
            timeout: Command timeout in seconds
            
        Returns:
            float: Reward value (0.0 or 1.0)
        """
        try:
            # Determine environment type based on env attributes
            if not ds:
                raise ValueError("Cannot find dataset (ds) in environment")
                
            # Check if it's swebench verified
            docker_image = ds["docker_image"] if "docker_image" in ds else ""
            swebench_verified = "sweb" in docker_image
            
            if swebench_verified:
                return self._calculate_reward_swebench_pod_manager(pod_manager, pod_name, ds, "/", timeout=300)
            else:
                return self._calculate_reward_r2e_pod_manager(pod_manager, pod_name, ds, "/root", timeout=300)
                
        except Exception as e:
            print(f"[RewardLogs] Error calculating reward: {e}")
            return 0.0

    def _calculate_reward_swebench_pod_manager(self, pod_manager, pod_name, ds, alt_path, timeout):
        """Calculate reward for SWE-bench verified environments using PodManager"""
        try:
            # Import required modules
            from swebench.harness.constants import (
                KEY_INSTANCE_ID, FAIL_TO_PASS, PASS_TO_PASS
            )
            from swebench.harness.grading import get_eval_tests_report, get_resolution_status
            from swebench.harness.log_parsers import MAP_REPO_TO_PARSER, get_eval_type
            from swebench.harness.constants import ResolvedStatus
            
            from swebench.harness.test_spec.test_spec import make_test_spec, TestSpec
            # Run tests using PodManager
            output, exit_code = pod_manager.execute_command(pod_name, f"bash {alt_path}/run_tests.sh", timeout)
            
            test_spec = make_test_spec(ds)
            # Get test spec from environment
            if not test_spec:
                print("[RewardLogs] Warning: No test_spec found, returning 0.0")
                return 0.0
                
            # Parse logs using swebench parser
            eval_status_map, found = self._get_logs_eval_kodo(test_spec, output)
            if not found:
                return 0.0
                
            eval_ref = {
                KEY_INSTANCE_ID: test_spec.instance_id,
                FAIL_TO_PASS: test_spec.FAIL_TO_PASS,
                PASS_TO_PASS: test_spec.PASS_TO_PASS,
            }
            
            report = get_eval_tests_report(
                eval_status_map, eval_ref, eval_type=get_eval_type(test_spec)
            )
            success = get_resolution_status(report) == ResolvedStatus.FULL.value
            
            print(f"[RewardLogs] SWE-bench reward calculation: success={success}")
            return int(success)
            
        except Exception as e:
            print(f"[RewardLogs] Error in _calculate_reward_swebench_pod_manager: {e}")
            return 0.0

    def _calculate_reward_r2e_pod_manager(self, pod_manager, pod_name, ds, alt_path, timeout):
        """Calculate reward for R2E environments using PodManager"""
        try:
            # Run tests using PodManager
            output, exit_code = pod_manager.execute_command(pod_name, f"timeout {timeout} bash {alt_path}/run_tests.sh", timeout)
            
            # Parse logs
            parse = self._parse_logs_kodo(ds, output)
            if not parse:
                print(f"[RewardLogs] No parse found in output, returning 0.0")
                return 0.0
                
            # Remove color codes from keys
            parse = self._decolor_dict_keys(parse)
            
            # Get expected output
            try:
                expected_json = ds.get("expected_output_json", "")
                if not expected_json:
                    print(f"[RewardLogs] No expected output json found in dataset, trying to read from file")
                    # Try to read from file
                    expected_output, _ = pod_manager.execute_command(pod_name, "cat /testbed/expected_test_output.json")
                    expected_json = expected_output
            except Exception as e:
                print(f"[RewardLogs] Error getting expected output: {e}")
                return 0.0
            
            expected = json.loads(expected_json)
            expected = self._decolor_dict_keys(expected)
#             print(f"[RewardLogs] pod name: {pod_name}, image: {ds.get('docker_image', '')}, Expected output: {expected}")
#             print(f"[RewardLogs] pod name: {pod_name}, image: {ds.get('docker_image', '')}, Parse output: {parse}")
            # Process keys
            parse = {k.split(" - ")[0]: parse[k] for k in sorted(parse.keys())}
            expected = {k.split(" - ")[0]: expected[k] for k in sorted(expected.keys())}
            
            # Compare results
            if len(parse) != len(expected):
                reward = 0.0
            else:
                # If ANY mismatch, reward = 0.0, else = 1.0
                match = True
                for k in parse.keys():
                    if not k:
                        continue
                    if k not in expected:
                        match = False
                        break
                    if parse[k] != expected[k]:
                        match = False
                        break
                reward = 1.0 if match else 0.0
                
            print(f"[RewardLogs] R2E reward calculation: reward={reward}")
            return reward
            
        except Exception as e:
            print(f"[RewardLogs] Error in _calculate_reward_r2e_pod_manager: {e}")
            return 0.0

    def _get_logs_eval_kodo(self, test_spec, content):
        """Evaluate logs for SWE-bench using kodo output"""
        try:
            from swebench.harness.constants import (
                APPLY_PATCH_FAIL, RESET_FAILED, TESTS_ERROR, TESTS_TIMEOUT
            )
            from swebench.harness.log_parsers import MAP_REPO_TO_PARSER
            from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS
            
            repo = test_spec.repo
            version = test_spec.version
            log_parser = MAP_REPO_TO_PARSER[repo]
            test_cmd = MAP_REPO_VERSION_TO_SPECS[repo][version]["test_cmd"]
            if isinstance(test_cmd, list):
                test_cmd = test_cmd[-1]

            # Check for bad codes
            bad_codes = list(
                filter(
                    lambda x: x in content,
                    [APPLY_PATCH_FAIL, RESET_FAILED, TESTS_ERROR, TESTS_TIMEOUT],
                )
            )
            if bad_codes:
                print(f"[RewardLogs] Bad code found in log: {bad_codes}")
                return {}, False

            # Get status map of evaluation results
            content = content.split(test_cmd)[-1]
            return log_parser(content, test_spec), True
            
        except Exception as e:
            print(f"[RewardLogs] Error in _get_logs_eval_kodo: {e}")
            return {}, False

    def _parse_logs_kodo(self, ds, log_output):
        """Parse logs based on dataset type"""
        try:
            # Import the parse function from the original docker.py logic
            repo_name = ds.get("repo", ds.get("repo_name", ""))
            
            # Try to import the actual parse_log_fn from the r2egym package
            try:
                from recipe.qianfan_swe.trainer.utils.execution_log_parser import parse_log_fn
                parser_func = parse_log_fn(repo_name)
                return parser_func(log_output)
            except ImportError:
                print("[RewardLogs] Warning: Could not import parse_log_fn, using fallback parsing")
                # Fallback: basic pytest parsing
                raise Exception(f"[RewardLogs] Could not import parse_log_fn, using fallback parsing")
            
        except Exception as e:
            print(f"[RewardLogs] Error parsing logs: {e}")
            return {}

    def _decolor_dict_keys(self, d):
        """Remove color codes from dictionary keys"""
        import re
        if not isinstance(d, dict):
            return d
        
        result = {}
        for k, v in d.items():
            # Remove ANSI color codes
            clean_key = re.sub(r'\x1b\[[0-9;]*m', '', str(k))
            result[clean_key] = v
        return result


class AsyncAgentExecutionEngine(AgentExecutionEngine):
    """
    Asynchronous Agent Execution Engine for SWE Training Framework.
    
    This class extends the base AgentExecutionEngine to provide asynchronous execution
    capabilities for running software engineering agents in a distributed training
    environment. It inherits all the core functionality from the parent class while
    adding async-specific optimizations and handling.
    
    Key Features:
    - Asynchronous agent trajectory execution
    - Concurrent model response generation
    - Non-blocking container and pod management
    - Async-safe tool execution and result collection
    - Enhanced performance monitoring for async operations
    
    The async engine is designed to work seamlessly with the existing synchronous
    components while providing improved throughput and resource utilization for
    large-scale training workflows.
    """
    def __init__(self, *args, **kwargs):
        """
        Initialize the AsyncAgentExecutionEngine.
        
        This constructor extends the base AgentExecutionEngine initialization to support
        asynchronous operations while maintaining compatibility with all existing
        configuration options.
        
        Args:
            *args: Variable length argument list passed to parent AgentExecutionEngine
            **kwargs: Arbitrary keyword arguments passed to parent AgentExecutionEngine.
                     All parameters from the parent class are supported, including:
                     - engine_name (str): Engine type, defaults to "verl"
                     - tokenizer: Tokenizer instance for text processing
                     - rollout_engine: Engine for model rollouts
                     - chat_parser: Parser for chat message formatting
                     - n_parallel_agents (int): Number of parallel agents
                     - trajectory_timeout: Timeout for trajectory execution
                     - gamma (float): Discount factor for rewards
                     - api_retries (int): Number of API retry attempts
                     - retry_limit (int): Maximum retry attempts
                     - max_steps (int): Maximum steps per trajectory
                     - max_response_length (int): Maximum response token length
                     - max_prompt_length (int): Maximum prompt token length
                     - config: Configuration object
                     - rollout_engine_args (dict): Arguments for rollout engine
                     - env_args (dict): Environment configuration arguments
                     - max_workers (int): Maximum thread pool workers
                     - enforce_max_prompt_length (bool): Enable prompt length enforcement
                     - overlong_filter (bool): Filter overlong trajectories
                     - logger: Logger instance for debugging
                     - extra_infos: Additional information for agents
                     - raw_prompts: Raw prompt data
                     - pod_managers: List of PodManager instances for Kubernetes
        
        Note:
            The async engine inherits all functionality from the base class and adds
            asynchronous execution capabilities without changing the initialization
            interface. All existing configuration options remain fully supported.
        """
        super().__init__(*args, **kwargs)
