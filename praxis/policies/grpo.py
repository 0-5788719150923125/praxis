"""
Group Relative Policy Optimization (GRPO) for outcome-based RL.

Based on DeepSeekMath paper (https://arxiv.org/abs/2402.03300).

Key insights from DeepSeek-R1:
- Process-level rewards made things worse, so we use outcome-level only
- Static reward functions (like solve_rate) work better than learned reward models
- Group-relative advantages provide variance reduction without a value network
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GRPO(nn.Module):
    """
    Group Relative Policy Optimization module.
    
    Key features:
    - No value network (memory efficient)
    - Outcome-level rewards only (same reward for entire sequence)
    - Group-based advantage estimation for variance reduction
    - KL divergence regularization to prevent reward hacking
    
    Implementation notes:
    - Uses static rewards from dataset (solve_rate) rather than learned reward model
    - Applies same reward to all tokens in a sequence (outcome-level)
    - Group normalization provides implicit baseline without critic network
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        
        # GRPO hyperparameters from DeepSeekMath paper
        self.group_size = getattr(config, "grpo_group_size", 64)  # G: samples per question
        self.kl_coeff = getattr(config, "grpo_kl_coeff", 0.04)   # β: KL penalty coefficient
        self.clip_ratio = getattr(config, "grpo_clip_ratio", 0.2)  # ε: PPO-style clipping
        
        # No policy network needed - GRPO works directly with LM hidden states
        # The paper doesn't use an additional policy network
        
        # Logging
        self.step_count = 0
        self.log_interval = 100

    def compute_group_advantages(self, rewards: torch.Tensor, group_size: int) -> torch.Tensor:
        """
        Compute group-relative advantages as per DeepSeekMath paper.
        
        For each group of G outputs for the same question:
        - Compute mean and std of rewards within the group
        - Normalize rewards to get advantages
        
        Args:
            rewards: Reward tensor [batch_size] where batch contains groups
            group_size: Number of outputs per question (G)
            
        Returns:
            advantages: Normalized advantages [batch_size]
        """
        batch_size = rewards.shape[0]
        num_groups = batch_size // group_size
        
        # Reshape to separate groups
        rewards_grouped = rewards.view(num_groups, group_size)
        
        # Compute per-group statistics
        group_means = rewards_grouped.mean(dim=1, keepdim=True)
        group_stds = rewards_grouped.std(dim=1, keepdim=True, unbiased=False)
        
        # Handle zero variance case (all rewards in group are identical)
        # When std is 0, all advantages should be 0 (no preference)
        has_variance = group_stds > 1e-8
        
        # Add minimum std to prevent division by zero
        min_std = 1e-3
        group_stds = torch.where(has_variance, group_stds, torch.tensor(min_std, device=group_stds.device))
        
        # Normalize within each group
        advantages_grouped = torch.where(
            has_variance.expand_as(rewards_grouped),
            (rewards_grouped - group_means) / group_stds,
            torch.zeros_like(rewards_grouped)  # Zero advantages when no variance
        )
        
        # Flatten back to batch
        advantages = advantages_grouped.view(batch_size)
        
        # Clamp advantages to prevent extreme values
        advantages = torch.clamp(advantages, min=-5.0, max=5.0)
        
        return advantages

    def forward(
        self,
        hidden_states: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor,
        rewards: Optional[torch.Tensor] = None,
        ref_logits: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass of GRPO with outcome-level rewards.

        Args:
            hidden_states: Hidden states from LM [batch_size, seq_len, hidden_size]
            logits: Current policy logits [batch_size, seq_len, vocab_size]
            labels: Token labels [batch_size, seq_len]
            rewards: Outcome-level rewards [batch_size] - same reward for entire sequence
            ref_logits: Reference model logits for KL [batch_size, seq_len, vocab_size]
            mask: Attention mask [batch_size, seq_len]

        Returns:
            hidden_states: Unchanged hidden states (GRPO doesn't modify them)
            losses: Dict containing grpo_loss and component losses
            
        Note: Each reward applies to the entire sequence (outcome-level), not individual tokens.
        """
        if rewards is None or not self.training:
            return hidden_states, None
            
        batch_size, seq_len, vocab_size = logits.shape
        actual_seq_len = seq_len - 1  # After shifting
        
        # Determine actual group size based on batch size
        if batch_size < self.group_size:
            # For small batches, treat entire batch as one group
            actual_group_size = batch_size
            if self.step_count == 0:
                print(f"[GRPO] Batch size {batch_size} < group size {self.group_size}, using batch as single group")
        elif batch_size % self.group_size != 0:
            # Find the largest divisor of batch_size that's <= group_size
            actual_group_size = 1
            for divisor in range(2, min(self.group_size, batch_size) + 1):
                if batch_size % divisor == 0:
                    actual_group_size = divisor
            if self.step_count == 0:
                print(f"[GRPO] Batch size {batch_size} not divisible by {self.group_size}, using group size {actual_group_size}")
        else:
            actual_group_size = self.group_size
        
        # Check rewards for validity
        if torch.isnan(rewards).any() or torch.isinf(rewards).any():
            print(f"[GRPO Warning] Invalid rewards detected: {rewards}")
            return hidden_states, None
        
        # Warn if all rewards are zero (common in early training)
        if torch.all(rewards == 0):
            if self.step_count % self.log_interval == 0:
                print(f"[GRPO Info] All rewards are zero in this batch - skipping RL loss")
            # Return None to skip RL loss when there's no learning signal
            return hidden_states, None
            
        # Compute group-relative advantages
        advantages = self.compute_group_advantages(rewards, actual_group_size)
        
        # Debug: Check input shapes (only on first step)
        if self.step_count == 0:
            print(f"\n[GRPO Debug] Input shapes:")
            print(f"  logits: {logits.shape}")
            print(f"  labels: {labels.shape}")
            print(f"  rewards: {rewards.shape}, values: {rewards}")
            print(f"  advantages: min={advantages.min():.4f}, max={advantages.max():.4f}, mean={advantages.mean():.4f}")
            print(f"  batch_size: {batch_size}, seq_len: {seq_len}, vocab_size: {vocab_size}")
        
        # Shift labels and logits for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        
        if ref_logits is not None:
            shift_ref_logits = ref_logits[:, :-1, :].contiguous()
        
        # Create mask for valid tokens
        if mask is not None:
            shift_mask = mask[:, 1:].contiguous()
        else:
            shift_mask = (shift_labels != -100).float()
        
        # Flatten for loss computation
        shift_logits_flat = shift_logits.view(-1, vocab_size)
        shift_labels_flat = shift_labels.view(-1)
        shift_mask_flat = shift_mask.view(-1)
        
        # Debug: Check flattened shapes
        if self.step_count == 0:
            print(f"[GRPO Debug] Flattened shapes:")
            print(f"  shift_logits_flat: {shift_logits_flat.shape}")
            print(f"  shift_labels_flat: {shift_labels_flat.shape}")
            print(f"  Expected size: {batch_size * actual_seq_len} = {batch_size} * {actual_seq_len}")
            print(f"  Actual flat size: {shift_labels_flat.numel()}")
        
        # Verify dimensions match
        if shift_labels_flat.numel() != batch_size * actual_seq_len:
            # Adjust actual_seq_len based on the real data
            actual_seq_len = shift_labels_flat.numel() // batch_size
            if self.step_count == 0:
                print(f"[GRPO Debug] Adjusted actual_seq_len to {actual_seq_len}")
        
        # Get log probabilities for the actual tokens
        log_probs = F.log_softmax(shift_logits_flat, dim=-1)
        
        # Replace -100 labels with 0 for gather (will be masked out anyway)
        gather_labels = shift_labels_flat.clone()
        gather_labels[gather_labels == -100] = 0
        
        token_log_probs = log_probs.gather(1, gather_labels.unsqueeze(1)).squeeze(1)
        
        # Apply mask to zero out padded positions
        token_log_probs = token_log_probs * shift_mask_flat
        
        # Reshape back to [batch_size, actual_seq_len]
        token_log_probs = token_log_probs.view(batch_size, actual_seq_len)
        
        # Compute per-sequence log probability (mean over valid tokens)
        seq_log_probs = token_log_probs.sum(dim=1) / shift_mask.sum(dim=1).clamp(min=1)
        
        # GRPO objective: log_prob * advantage
        # Note: advantages are outcome-level (same for entire sequence)
        policy_term = seq_log_probs * advantages
        
        # Check for NaN/Inf before computing loss
        if torch.isnan(policy_term).any() or torch.isinf(policy_term).any():
            print(f"[GRPO Warning] NaN/Inf detected in policy term")
            print(f"  seq_log_probs: min={seq_log_probs.min():.4f}, max={seq_log_probs.max():.4f}")
            print(f"  advantages: min={advantages.min():.4f}, max={advantages.max():.4f}")
            # Return zero loss to skip this batch
            return hidden_states, None
            
        policy_loss = -policy_term.mean()
        
        # KL divergence term
        kl_loss = 0.0
        if ref_logits is not None:
            # Compute KL divergence following the paper's unbiased estimator
            shift_ref_logits_flat = shift_ref_logits.view(-1, vocab_size)
            
            ref_log_probs = F.log_softmax(shift_ref_logits_flat, dim=-1)
            ref_token_log_probs = ref_log_probs.gather(1, gather_labels.unsqueeze(1)).squeeze(1)
            
            # Apply mask to reference token log probs too
            ref_token_log_probs = ref_token_log_probs * shift_mask_flat
            
            # Unbiased KL estimator: π_ref/π_θ - log(π_ref/π_θ) - 1
            # Compute log ratio using flat tensors first
            token_log_probs_flat = token_log_probs.view(-1)
            log_ratio = ref_token_log_probs - token_log_probs_flat
            
            # Apply mask and clamp to prevent numerical issues
            log_ratio = log_ratio * shift_mask_flat
            log_ratio = torch.clamp(log_ratio, min=-5.0, max=5.0)
            
            # Compute ratio with clamping to prevent overflow
            ratio = torch.exp(log_ratio)
            ratio = torch.clamp(ratio, max=100.0)  # Prevent extreme values
            
            # Per-token KL: only compute for valid tokens
            # Note: For masked positions, all terms are 0
            token_kl = (ratio - log_ratio - 1) * shift_mask_flat
            
            # Average KL per sequence
            token_kl = token_kl.view(batch_size, actual_seq_len)
            seq_kl = token_kl.sum(dim=1) / shift_mask.sum(dim=1).clamp(min=1)
            
            kl_loss = seq_kl.mean()
        
        # Total GRPO loss
        total_loss = policy_loss + self.kl_coeff * kl_loss
        
        # Logging
        self.step_count += 1
        if self.step_count % self.log_interval == 0:
            with torch.no_grad():
                print(f"\n[GRPO] Step {self.step_count}:")
                print(f"  Group size: {actual_group_size} (outcome-level rewards)")
                print(f"  Rewards: mean={rewards.mean():.3f}, std={rewards.std():.3f}")
                print(f"  Advantages: mean={advantages.mean():.3f}, std={advantages.std():.3f}")
                print(f"  Policy loss: {policy_loss:.4f}")
                print(f"  KL loss: {kl_loss:.4f}")
                print(f"  Total loss: {total_loss:.4f}")
                if actual_group_size < self.group_size:
                    print(f"  Warning: Batch size should be divisible by {self.group_size}")
        
        losses = {
            "grpo_loss": total_loss,
            "policy_loss": policy_loss,
            "kl_loss": kl_loss,
            "mean_reward": rewards.mean(),
            "mean_advantage": advantages.mean(),
        }
        
        # Return unchanged hidden states and losses
        return hidden_states, losses