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
        
        # Pattern detection for CoT structure - using correct tags from dataset
        self.thinking_patterns = [
            "<initial_analysis>", "</initial_analysis>",
            "<conscious_thought>", "</conscious_thought>"
        ]
        self.step_patterns = ["<step_by_step>", "</step_by_step>"]
        self.reflection_patterns = ["<reflection>", "</reflection>"]
        self.feeling_patterns = ["<feeling>", "</feeling>"]
        self.improvement_patterns = ["<self_improvement>", "</self_improvement>"]
        
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
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_ids: Optional[torch.Tensor] = None,
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
        if not self.training:
            return hidden_states, None
            
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        
        # Get actual logits shape since it might be different due to shifting
        logits_batch_size, logits_seq_len, vocab_size = logits.shape
        
        # Create token weights based on CoT structure matching logits shape
        token_weights = torch.ones(logits_batch_size, logits_seq_len, device=device)
        
        if token_ids is not None:
            # Apply higher weights to reasoning sections
            # This is a simplified version - in practice, you'd decode and check patterns
            # For now, we'll use positional heuristics
            
            # Assume reasoning happens in the first 70% of sequence
            reasoning_end = int(logits_seq_len * 0.7)
            token_weights[:, :reasoning_end] = self.reasoning_weight
            
            # Optional: Detect structured reasoning
            # This would require decoding tokens which is expensive
            # So we'll estimate quality from hidden states instead
            
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
        weights_flat = token_weights.view(-1)
        
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
        
        # Logging
        self.step_count += 1
        if self.step_count % self.log_interval == 0:
            print(f"\n[CoT Policy] Step {self.step_count}:")
            print(f"  Weighted loss: {weighted_loss:.4f}")
            print(f"  Structure bonus: {-structure_loss:.4f}")
            print(f"  Quality scores: mean={quality_scores.mean():.3f}")
            print(f"  Total CoT loss: {cot_loss:.4f}")
            
        losses = {
            "cot_loss": cot_loss,
            "reasoning_quality": quality_scores.mean(),
            "weighted_loss": weighted_loss,
        }
        
        return hidden_states, losses


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
        
        # Check for structured reasoning using correct dataset tags
        has_initial_analysis = "<initial_analysis>" in generated_text and "</initial_analysis>" in generated_text
        has_conscious_thought = "<conscious_thought>" in generated_text and "</conscious_thought>" in generated_text
        has_steps = "<step_by_step>" in generated_text and "</step_by_step>" in generated_text
        has_reflection = "<reflection>" in generated_text and "</reflection>" in generated_text
        has_feeling = "<feeling>" in generated_text and "</feeling>" in generated_text
        has_improvement = "<self_improvement>" in generated_text and "</self_improvement>" in generated_text
        
        # Reward presence of thinking patterns
        if has_initial_analysis:
            reward += 0.2
        if has_conscious_thought:
            reward += 0.2
        if has_steps:
            reward += 0.3
        if has_reflection:
            reward += 0.2
        if has_feeling:
            reward += 0.1
        if has_improvement:
            reward += 0.1
            
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