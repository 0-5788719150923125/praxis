import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    def __init__(self, penalty_weight=0):
        super().__init__()
        self.penalty_weight = penalty_weight

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        input_ids: torch.Tensor,
        *args,
        **kwargs,
    ):
        shift_logits = logits[..., :-1, :]
        shift_logits = shift_logits.reshape(-1, shift_logits.shape[-1])
        shift_labels = labels[..., 1:].reshape(-1)
        ce_loss = F.cross_entropy(
            shift_logits, shift_labels, reduction="none", ignore_index=-100
        )
        if self.penalty_weight == 0:
            return ce_loss.mean()
        token_output = torch.argmax(shift_logits, dim=1)
        duplicated_masks = (
            torch.eq(input_ids.view(-1), token_output.unsqueeze(-1)).any(dim=-1).float()
        )
        loss = ce_loss * (1 + duplicated_masks * self.penalty_weight)
        return loss.mean()


def test_dedup_loss():
    # Setup
    vocab_size = 1000
    batch_size = 2
    seq_length = 5

    # Create sample data
    # Logits shape: [batch_size * seq_length, vocab_size]
    logits = torch.randn(batch_size * seq_length, vocab_size)

    # Labels shape: [batch_size * seq_length]
    labels = torch.randint(0, vocab_size, (batch_size * seq_length,))

    # Input tokens shape: [batch_size * seq_length, input_seq_length]
    # Representing the context window for each position
    input_seq_length = 10
    input_tokens = torch.randint(
        0, vocab_size, (batch_size * seq_length, input_seq_length)
    )

    # Create loss function
    criterion = CrossEntropyLoss(penalty_weight=1.0)

    # Test case 1: Normal case
    loss1 = criterion(logits, labels, input_tokens)
    print(f"Test 1 - Normal case loss: {loss1.item()}")

    # Test case 2: With duplicates
    # Force some predictions to match input tokens
    with torch.no_grad():
        forced_token = input_tokens[0, 0]  # Take first token from input
        logits[0, :] = -100  # Set all logits to very negative value
        logits[0, forced_token] = 100  # Make the model surely predict the input token

    loss2 = criterion(logits, labels, input_tokens)
    print(f"Test 2 - With forced duplicate loss: {loss2.item()}")

    # Test case 3: No penalty
    criterion_no_penalty = CrossEntropyLoss(penalty_weight=0.0)
    loss3 = criterion_no_penalty(logits, labels, input_tokens)
    print(f"Test 3 - No penalty loss: {loss3.item()}")

    # Verify that loss with duplicates is higher than normal case
    assert loss2 > loss1, "Loss with duplicates should be higher than normal case"

    print("All tests passed!")


if __name__ == "__main__":
    test_dedup_loss()
