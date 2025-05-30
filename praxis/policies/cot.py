"""
Chain of Thought (CoT) policy for training models with step-by-step reasoning.

This module implements a simple CoT training approach that:
1. Encourages generation of reasoning steps before final answers
2. Rewards correct reasoning patterns
3. Uses supervised learning with structured prompts
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChainOfThought(nn.Module):
    """
    Chain of Thought training policy.
    
    This is a simplified approach that:
    - Uses supervised learning on CoT examples
    - Rewards structured reasoning (presence of thinking tags)
    - Applies higher weight to reasoning steps vs final answer
    
    Key features:
    - No complex RL needed initially - supervised learning works well for CoT
    - Gradually introduces rewards for reasoning structure
    - Can be extended with REINFORCE for more complex scenarios
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        
        # CoT-specific parameters (keeping them local as requested)
        self.reasoning_weight = 1.5  # Higher weight for reasoning tokens
        self.structure_bonus = 0.2   # Bonus for proper tag structure
        self.min_reasoning_length = 50  # Minimum tokens for reasoning
        
        # CoT policy is now simple - just applies pre-computed token weights from builder
        
        # Optional: Simple MLP to predict reasoning quality
        self.quality_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # Logging
        self.step_count = 0
        self.log_interval = 100
        
        # CoT tracking statistics
        self.cot_stats = {
            "total_batches": 0,
            "batches_with_cot": 0,
            "total_cot_tokens": 0,
            "total_tokens": 0,
        }
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass for CoT training.
        
        For now, we primarily return the hidden states unchanged and compute
        a weighted loss that emphasizes reasoning steps. This can be extended
        with REINFORCE later if needed.
        
        Args:
            hidden_states: Hidden states from LM [batch_size, seq_len, hidden_size]
            logits: Model logits [batch_size, seq_len, vocab_size]
            labels: Token labels [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            token_ids: Original token IDs for pattern detection [batch_size, seq_len]
            
        Returns:
            hidden_states: Unchanged hidden states
            losses: Dict containing cot_loss and metrics
        """
        # Metadata contains pre-computed token weights from the data pipeline
        if not self.training:
            return hidden_states, None
            
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        
        # Get actual logits shape since it might be different due to shifting
        logits_batch_size, logits_seq_len, vocab_size = logits.shape
        
        # Create token weights - use pre-computed weights if available, otherwise default
        if token_weights is not None:
            # Use pre-computed token weights from builder
            # Ensure shape matches [batch_size, seq_len] for logits
            if token_weights.dim() == 1:
                # Single sequence, broadcast to batch
                weights = token_weights.unsqueeze(0).expand(logits_batch_size, -1)
            else:
                weights = token_weights
            
            # Ensure weights match the logits sequence length (accounting for shift)
            if weights.shape[-1] > logits_seq_len:
                weights = weights[..., :logits_seq_len]
            elif weights.shape[-1] < logits_seq_len:
                # Pad with default weight if needed
                padding = torch.ones(
                    *weights.shape[:-1], 
                    logits_seq_len - weights.shape[-1], 
                    device=device
                )
                weights = torch.cat([weights.to(device), padding], dim=-1)
            
            token_weights = weights
            
            # Detailed logging for validation
            if self.step_count % self.log_interval == 0:
                non_default_tokens = (token_weights != 1.0).sum().item()
                print(f"[CoT Policy] Using pre-computed token weights:")
                print(f"  Shape: {token_weights.shape}")
                print(f"  Tokens with non-default weights: {non_default_tokens}/{token_weights.numel()}")
                print(f"  Weight range: [{token_weights.min():.3f}, {token_weights.max():.3f}]")
                print(f"  Mean weight: {token_weights.mean():.3f}")
                
                # Show distribution of weights
                unique_weights = torch.unique(token_weights)
                print(f"  Unique weights: {unique_weights.tolist()}")
                
                # Show per-sequence stats
                for batch_idx in range(min(2, token_weights.shape[0])):  # Log first 2 sequences
                    seq_weights = token_weights[batch_idx]
                    non_default = (seq_weights != 1.0).sum().item()
                    print(f"  Sequence {batch_idx}: {non_default}/{len(seq_weights)} non-default tokens")
                    
                    # Validate that high-weight tokens actually contain CoT content
                    if non_default > 0:
                        self._validate_token_weights(batch_idx, seq_weights, labels, logits)
        else:
            # Default to uniform weights with slight emphasis on early tokens (reasoning)
            token_weights = torch.ones(logits_batch_size, logits_seq_len, device=device)
            reasoning_end = int(logits_seq_len * 0.7)
            token_weights[:, :reasoning_end] = self.reasoning_weight
            
        # Compute reasoning quality score
        if attention_mask is not None:
            # Pool hidden states for quality estimation
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states)
            sum_hidden = (hidden_states * mask_expanded).sum(dim=1)
            count = attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
            pooled_hidden = sum_hidden / count
        else:
            pooled_hidden = hidden_states.mean(dim=1)
            
        quality_scores = self.quality_head(pooled_hidden).squeeze(-1)
        
        # Compute weighted cross-entropy loss
        # The logits passed here are already shifted by the model (:-1)
        # So labels should already match the logits shape
        
        if attention_mask is not None:
            # Also need to trim attention mask to match shifted logits
            attention_mask_shifted = attention_mask[..., :-1] if attention_mask.shape[-1] > logits_seq_len else attention_mask
            token_weights = token_weights * attention_mask_shifted
            
        # Flatten for loss computation
        batch_size_actual, seq_len_actual, vocab_size = logits.shape
        
        # Debug shapes
        # print(f"[CoT Debug] logits shape: {logits.shape}")
        # print(f"[CoT Debug] labels shape: {labels.shape}")
        # print(f"[CoT Debug] token_weights shape: {token_weights.shape}")
        
        # Ensure labels match the shifted logits shape
        if labels.shape[-1] != seq_len_actual:
            # This shouldn't happen if everything is set up correctly
            print(f"[CoT Warning] Label shape mismatch: labels {labels.shape} vs logits {logits.shape}")
            # Trim labels to match
            labels = labels[..., :seq_len_actual].contiguous()
        
        logits_flat = logits.view(-1, vocab_size)
        labels_flat = labels.view(-1)
        weights_flat = token_weights.contiguous().view(-1)
        
        # Filter out padding tokens (label == -100)
        valid_mask = labels_flat != -100
        
        if valid_mask.sum() == 0:
            # No valid tokens to compute loss on
            return hidden_states, None
            
        # Compute per-token loss only on valid tokens
        per_token_loss = F.cross_entropy(
            logits_flat,
            labels_flat,
            reduction='none',
            ignore_index=-100
        )
        
        # Apply weights and average
        weighted_loss = (per_token_loss * weights_flat).sum() / weights_flat[valid_mask].sum().clamp(min=1)
        
        # Add structure bonus based on quality scores
        structure_loss = -self.structure_bonus * quality_scores.mean()
        
        # Total CoT loss
        cot_loss = weighted_loss + structure_loss
        
        # Update statistics
        self.cot_stats["total_batches"] += 1
        self.cot_stats["total_tokens"] += token_weights.numel()
        
        if token_weights is not None:
            cot_tokens_in_batch = (token_weights != 1.0).sum().item()
            if cot_tokens_in_batch > 0:
                self.cot_stats["batches_with_cot"] += 1
                self.cot_stats["total_cot_tokens"] += cot_tokens_in_batch
        
        # Logging
        self.step_count += 1
        if self.step_count % self.log_interval == 0:
            print(f"\n[CoT Policy] Step {self.step_count}:")
            print(f"  Weighted loss: {weighted_loss:.4f}")
            print(f"  Structure bonus: {-structure_loss:.4f}")
            print(f"  Quality scores: mean={quality_scores.mean():.3f}")
            print(f"  Total CoT loss: {cot_loss:.4f}")
            
            # Print CoT usage statistics
            total_batches = self.cot_stats["total_batches"]
            cot_batches = self.cot_stats["batches_with_cot"]
            total_tokens = self.cot_stats["total_tokens"]
            cot_tokens = self.cot_stats["total_cot_tokens"]
            
            cot_batch_pct = (cot_batches / total_batches * 100) if total_batches > 0 else 0
            cot_token_pct = (cot_tokens / total_tokens * 100) if total_tokens > 0 else 0
            
            print(f"  CoT Usage Stats:")
            print(f"    Batches with CoT: {cot_batches}/{total_batches} ({cot_batch_pct:.1f}%)")
            print(f"    CoT tokens: {cot_tokens}/{total_tokens} ({cot_token_pct:.2f}%)")
            print(f"    Avg CoT tokens per CoT batch: {cot_tokens/max(cot_batches,1):.1f}")
            
        losses = {
            "cot_loss": cot_loss,
            "reasoning_quality": quality_scores.mean(),
            "weighted_loss": weighted_loss,
        }
        
        return hidden_states, losses
    
    def _validate_token_weights(self, batch_idx, seq_weights, labels, logits):
        """
        Validate that tokens with high weights actually contain CoT content.
        This helps ensure our weight assignment is working correctly.
        """
        try:
            # Import here to avoid circular imports
            from builders import COT_TAGS
            
            # Get the labels for this sequence (these are the target tokens)
            if batch_idx < labels.shape[0]:
                seq_labels = labels[batch_idx]
                
                # Find tokens with weights > 1.0 (CoT tokens)
                high_weight_mask = seq_weights > 1.0
                high_weight_positions = torch.where(high_weight_mask)[0]
                
                if len(high_weight_positions) > 0:
                    # Sample a few positions to check
                    sample_size = min(10, len(high_weight_positions))
                    sample_positions = high_weight_positions[:sample_size]
                    
                    # Get the token IDs at these positions
                    sample_tokens = seq_labels[sample_positions]
                    
                    # Try to decode them (this is approximate since we don't have the tokenizer here)
                    # We'll just check if the weights are consistent with expected patterns
                    weights_at_samples = seq_weights[sample_positions]
                    
                    print(f"    Sample high-weight tokens (positions): {sample_positions.tolist()}")
                    print(f"    Sample weights: {weights_at_samples.tolist()}")
                    
                    # Check if weights are clustered (indicating tag regions)
                    consecutive_groups = self._find_consecutive_groups(high_weight_positions)
                    print(f"    Consecutive high-weight groups: {len(consecutive_groups)}")
                    for i, group in enumerate(consecutive_groups[:3]):  # Show first 3 groups
                        print(f"      Group {i}: positions {group[0]}-{group[-1]} (length: {len(group)})")
                        
        except Exception as e:
            print(f"    Validation error: {e}")
    
    def _find_consecutive_groups(self, positions):
        """Find groups of consecutive positions in the tensor."""
        if len(positions) == 0:
            return []
        
        groups = []
        current_group = [positions[0].item()]
        
        for i in range(1, len(positions)):
            if positions[i].item() == positions[i-1].item() + 1:
                current_group.append(positions[i].item())
            else:
                groups.append(current_group)
                current_group = [positions[i].item()]
        
        groups.append(current_group)
        return groups


class ChainOfThoughtREINFORCE(nn.Module):
    """
    Extended CoT policy using REINFORCE for more complex reasoning rewards.
    
    This extends the basic CoT approach with:
    - REINFORCE-based training for reasoning paths
    - Rewards for correct intermediate steps
    - Penalties for incorrect reasoning
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        
        # Use the existing REINFORCE implementation as base
        from praxis.policies.reinforce import REINFORCE
        self.reinforce = REINFORCE(config)
        
        # CoT-specific reward shaping
        self.reasoning_reward_weight = 2.0
        self.structure_reward = 0.5
        self.coherence_reward = 0.3
        
        # Load tags from centralized configuration
        if COT_TAGS is not None:
            self.cot_tags = COT_TAGS
            self.tag_rewards = COT_TAGS["tag_rewards"]
        else:
            # Fallback
            self.cot_tags = {
                "wrapper": {
                    "thinking": ("<thinking>", "</thinking>"),
                    "output": ("<output>", "</output>"),
                },
                "thinking_components": {
                    "initial_analysis": ("<initial_analysis>", "</initial_analysis>"),
                    "conscious_thought": ("<conscious_thought>", "</conscious_thought>"),
                    "step_by_step": ("<step_by_step>", "</step_by_step>"),
                    "reflection": ("<reflection>", "</reflection>"),
                    "feeling": ("<feeling>", "</feeling>"),
                    "self_improvement": ("<self_improvement>", "</self_improvement>"),
                    "subcomponent_analysis": ("<subcomponent_analysis>", "</subcomponent_analysis>"),
                },
            }
            self.tag_rewards = {tag: 0.2 for tag in list(self.cot_tags["wrapper"].keys()) + list(self.cot_tags["thinking_components"].keys())}
        
    def compute_reasoning_rewards(
        self,
        generated_text: str,
        ground_truth: Optional[str] = None
    ) -> float:
        """
        Compute rewards specifically for reasoning quality.
        
        Rewards:
        - Presence of thinking/step_by_step tags: +0.5
        - Logical flow (each step follows previous): +0.3
        - Reaching correct conclusion: +1.0
        - Length of reasoning (longer is better up to a point): +0.2
        """
        reward = 0.0
        
        # Check for structured reasoning using centralized tags
        # Check wrapper tags
        for tag_name, (open_tag, close_tag) in self.cot_tags["wrapper"].items():
            if open_tag in generated_text and close_tag in generated_text:
                reward += self.tag_rewards.get(tag_name, 0.2)
        
        # Check thinking component tags
        for tag_name, (open_tag, close_tag) in self.cot_tags["thinking_components"].items():
            if open_tag in generated_text and close_tag in generated_text:
                reward += self.tag_rewards.get(tag_name, 0.2)
            
        # Check reasoning length
        reasoning_length = len(generated_text.split())
        if reasoning_length > 50:
            reward += min(0.2, reasoning_length / 500)
            
        # If ground truth provided, check final answer
        if ground_truth and ground_truth.lower() in generated_text.lower():
            reward += 1.0
            
        return reward * self.reasoning_reward_weight
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor,
        rewards: Optional[torch.Tensor] = None,
        generated_texts: Optional[list] = None,
        ground_truths: Optional[list] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass using REINFORCE with CoT-specific rewards.
        """
        # If we have generated texts, compute CoT-specific rewards
        if generated_texts is not None and ground_truths is not None:
            cot_rewards = []
            for gen_text, gt in zip(generated_texts, ground_truths):
                reward = self.compute_reasoning_rewards(gen_text, gt)
                cot_rewards.append(reward)
            
            cot_rewards = torch.tensor(cot_rewards, device=hidden_states.device)
            
            # Combine with existing rewards if any
            if rewards is not None:
                rewards = rewards + cot_rewards
            else:
                rewards = cot_rewards
                
        # Use REINFORCE for policy gradient training
        policy_hidden, rl_loss = self.reinforce(
            hidden_states, rewards=rewards, mask=kwargs.get('attention_mask')
        )
        
        # Create losses dict with REINFORCE loss and CoT-specific metrics
        losses = {}
        if rl_loss is not None:
            losses['rl_loss'] = rl_loss
            
        # Add CoT-specific metrics
        if rewards is not None and rewards.numel() > 0:
            losses['cot_reward_mean'] = rewards.mean()
        else:
            losses['cot_reward_mean'] = torch.tensor(0.0, device=hidden_states.device)
            
        return policy_hidden, losses if losses else None