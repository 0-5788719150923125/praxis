"""Tests for the engagement-prediction reward (P2) and policy (P3)."""

import torch

from praxis import PraxisConfig
from praxis.policies import EngagementPolicy, needs_rl_datasets, normalize_rl_types
from praxis.policies.engagement_reward import (
    ENERGY_FLOOR,
    HomeostaticEnergy,
    activation,
    recall,
    response_energy,
)


class TestReward:
    def test_activation_fires_on_overlap(self):
        assert activation([1, 2, 3], [3, 9]) == 1.0
        assert activation([1, 2, 3], [7, 8]) == 0.0
        assert activation([], [1]) == 0.0

    def test_recall_is_graded(self):
        assert recall([1, 2, 3, 4], [1, 2, 9]) == 0.5  # 2 of 4 predicted mentioned
        assert recall([1, 2], [1, 2]) == 1.0
        assert recall([], [1]) == 0.0

    def test_response_energy_floors_on_engagement(self):
        # Any genuine response sustains energy regardless of prediction match;
        # quality lifts it to 1.0. No interaction -> no energy.
        assert response_energy(False) == 0.0
        assert response_energy(True, 0.0) == ENERGY_FLOOR
        assert response_energy(True, 1.0) == 1.0
        assert ENERGY_FLOOR < response_energy(True, 0.5) < 1.0


class TestHomeostaticEnergy:
    def test_accumulates_fast_then_satiates(self):
        e = HomeostaticEnergy()
        first = e.update(1.0)
        # Diminishing returns: each successive full activation adds less.
        gains = [first]
        for _ in range(5):
            before = e.value
            e.update(1.0)
            gains.append(e.value - before)
        assert gains[0] > 0
        assert all(gains[i] >= gains[i + 1] - 1e-9 for i in range(len(gains) - 1))

    def test_decays_without_activation(self):
        # Decay is wall-clock, not event-driven: energy depletes overnight even
        # if nothing ever calls update().
        t = [0.0]
        e = HomeostaticEnergy(init=0.5, clock=lambda: t[0])
        t[0] += e.half_life_s
        assert abs(e.value - 0.25) < 1e-9
        t[0] += 8 * e.half_life_s
        assert e.value < 0.01

    def test_energy_stays_bounded(self):
        e = HomeostaticEnergy()
        for _ in range(10_000):
            e.update(1.0)
        assert 0.0 <= e.value <= e.e_max


class TestEngagementPolicy:
    def _config(self):
        return PraxisConfig(hidden_size=64, dropout=0.0, rl_weight=0.1)

    def test_forward_produces_loss_and_metrics(self):
        policy = EngagementPolicy(self._config())
        policy.train()
        b, t, v = 4, 10, 50
        logits = torch.randn(b, t, v, requires_grad=True)
        labels = torch.randint(0, v, (b, t))
        mask = torch.zeros(b, t)
        mask[:, 5:] = 1.0  # last tokens are the "answer" region

        loss, metrics = policy(logits=logits, labels=labels, assistant_mask=mask)

        assert loss is not None and loss.requires_grad and torch.isfinite(loss)
        for key in (
            "engagement_energy",
            "engagement_activation_rate",
            "engagement_recall",
            "engagement_advantage",
        ):
            assert key in metrics
        loss.backward()
        assert logits.grad is not None

    def test_perfect_prediction_drives_energy_up(self):
        policy = EngagementPolicy(self._config())
        policy.train()
        b, t, v = 4, 8, 30
        labels = torch.randint(0, v, (b, t))
        mask = torch.ones(b, t)
        # Logits that argmax exactly to the labels -> full recall every step.
        logits = torch.full((b, t, v), -10.0)
        for i in range(b):
            for j in range(t):
                logits[i, j, labels[i, j]] = 10.0

        e_prev = 0.0
        for _ in range(5):
            _, metrics = policy(logits=logits, labels=labels, assistant_mask=mask)
            assert metrics["engagement_activation_rate"] == 1.0
            assert metrics["engagement_energy"] >= e_prev
            e_prev = metrics["engagement_energy"]
        assert e_prev > 0.0

    def test_noop_when_no_mask_or_eval(self):
        policy = EngagementPolicy(self._config())
        logits = torch.randn(2, 6, 20)
        labels = torch.randint(0, 20, (2, 6))
        policy.eval()
        assert policy(
            logits=logits, labels=labels, assistant_mask=torch.ones(2, 6)
        ) == (
            None,
            {},
        )
        policy.train()
        assert policy(logits=logits, labels=labels, assistant_mask=None) == (None, {})


def test_engagement_needs_no_rl_datasets():
    assert normalize_rl_types("engagement") == ["engagement"]
    assert needs_rl_datasets("engagement") is False
    # Coexists with a weight controller; neither pulls the RL collection.
    assert needs_rl_datasets("harmonic_weight_wave") is False


class TestLiveDrainCallback:
    """The training-loop seam: live web rewards -> policy energy baseline."""

    def _setup(self, period=1):
        import types

        from praxis.callbacks.lightning import EngagementLiveRewardCallback
        from praxis.policies.engagement_channel import LIVE_ENGAGEMENT

        LIVE_ENGAGEMENT.drain()  # start clean
        policy = EngagementPolicy(PraxisConfig(hidden_size=32, dropout=0.0))
        pl = types.SimpleNamespace(model=types.SimpleNamespace(policy=policy))
        trainer = types.SimpleNamespace(callback_metrics={})
        cb = EngagementLiveRewardCallback(period=period)
        return cb, trainer, pl, policy, LIVE_ENGAGEMENT

    def test_drain_folds_live_reward_into_energy(self):
        cb, trainer, pl, policy, channel = self._setup(period=1)
        assert policy.energy.value == 0.0
        channel.submit(["paris"], ["i", "think", "paris"])  # activation 1.0
        cb.on_train_batch_end(trainer, pl, None, None, 0)
        assert policy.energy.value > 0.0
        assert trainer.callback_metrics["engagement_live_count"].item() == 1.0
        assert channel.snapshot()["buffered"] == 0  # drained

    def test_respects_period(self):
        cb, trainer, pl, policy, channel = self._setup(period=3)
        channel.submit(["paris"], ["paris"])
        cb.on_train_batch_end(trainer, pl, None, None, 0)  # step 1: no drain
        assert policy.energy.value == 0.0
        cb.on_train_batch_end(trainer, pl, None, None, 1)  # step 2: no drain
        cb.on_train_batch_end(trainer, pl, None, None, 2)  # step 3: drains
        assert policy.energy.value > 0.0


class TestJokePolicy:
    """JokePolicy reuses EngagementPolicy's machinery under the joke namespace."""

    def test_emits_joke_namespaced_metrics(self):
        from praxis.policies import JokePolicy

        policy = JokePolicy(PraxisConfig(hidden_size=32, dropout=0.0))
        policy.train()
        b, t, v = 3, 8, 40
        logits = torch.randn(b, t, v, requires_grad=True)
        labels = torch.randint(0, v, (b, t))
        loss, metrics = policy(
            logits=logits, labels=labels, assistant_mask=torch.ones(b, t)
        )
        assert loss is not None and torch.isfinite(loss)
        assert set(metrics) == {
            "joke_energy",
            "joke_activation_rate",
            "joke_recall",
            "joke_reward",
            "joke_reward_baseline",
            "joke_advantage",
        }

    def test_live_approval_channel_drains_into_joke_policy(self):
        import types

        from praxis.callbacks.lightning import EngagementLiveRewardCallback
        from praxis.policies import JokePolicy
        from praxis.policies.engagement_channel import LIVE_JOKES

        LIVE_JOKES.drain()
        policy = JokePolicy(PraxisConfig(hidden_size=32, dropout=0.0))
        pl = types.SimpleNamespace(model=types.SimpleNamespace(policy=policy))
        trainer = types.SimpleNamespace(callback_metrics={})
        cb = EngagementLiveRewardCallback(
            period=1,
            channel=LIVE_JOKES,
            policy_class_name="JokePolicy",
            metric_prefix="joke",
        )
        LIVE_JOKES.submit_scalar(1.0)  # a human approved a joke
        cb.on_train_batch_end(trainer, pl, None, None, 0)
        assert policy.energy.value > 0.0
        assert trainer.callback_metrics["joke_live_count"].item() == 1.0


class TestTaskScoping:
    """Coexisting recall policies must partition the task space - identical
    metrics across engagement and joke was the bug this guards against."""

    def _config(self):
        return PraxisConfig(hidden_size=64, dropout=0.0, rl_weight=0.1)

    def _batch(self, seed=0):
        torch.manual_seed(seed)
        from praxis.tasks import TaskType

        b, t, v = 4, 10, 50
        logits = torch.randn(b, t, v, requires_grad=True)
        labels = torch.randint(0, v, (b, t))
        mask = torch.zeros(b, t)
        mask[:, 5:] = 1.0
        # Rows 0-1 are conversation, rows 2-3 are jokes.
        task_ids = torch.full((b, t), int(TaskType.CONVERSATION))
        task_ids[2:] = int(TaskType.JOKE)
        return logits, labels, mask, task_ids

    def test_policies_score_disjoint_rows(self):
        from praxis.policies import EngagementPolicy, JokePolicy

        logits, labels, mask, task_ids = self._batch()
        # Conversation rows get perfect recall (labels = the model's argmax);
        # joke rows keep random labels. Disjoint scoping must show the split.
        labels[:2, 1:] = logits.argmax(dim=-1)[:2, :-1]  # next-token aligned
        eng, joke = EngagementPolicy(self._config()), JokePolicy(self._config())
        eng.train(), joke.train()
        _, em = eng(logits=logits, labels=labels, assistant_mask=mask, task_type_ids=task_ids)
        _, jm = joke(logits=logits, labels=labels, assistant_mask=mask, task_type_ids=task_ids)
        assert em and jm
        assert em["engagement_recall"] == 1.0
        assert jm["joke_recall"] < 1.0

    def test_policy_noops_when_no_rows_match(self):
        from praxis.policies import JokePolicy
        from praxis.tasks import TaskType

        logits, labels, mask, task_ids = self._batch()
        task_ids.fill_(int(TaskType.CONVERSATION))  # no joke rows
        joke = JokePolicy(self._config())
        joke.train()
        loss, metrics = joke(
            logits=logits, labels=labels, assistant_mask=mask, task_type_ids=task_ids
        )
        assert loss is None and metrics == {}

    def test_no_task_ids_falls_back_to_scoring_all(self):
        from praxis.policies import EngagementPolicy

        logits, labels, mask, _ = self._batch()
        pol = EngagementPolicy(self._config())
        pol.train()
        loss, metrics = pol(logits=logits, labels=labels, assistant_mask=mask)
        assert loss is not None and metrics


def test_difficulty_weighter_pads_old_checkpoints():
    """ema_loss saved before TaskType.JOKE existed loads with NaN padding."""
    import torch
    from praxis.tasks import TaskType
    from praxis.tasks.weighter import DifficultyTaskLossWeighter

    w = DifficultyTaskLossWeighter()
    sd = w.state_dict()
    sd["ema_loss"] = torch.ones(len(TaskType) - 1)  # pre-JOKE checkpoint
    w2 = DifficultyTaskLossWeighter()
    w2.load_state_dict(sd)
    assert w2.ema_loss.numel() == len(TaskType)
    assert torch.isnan(w2.ema_loss[-1])
    assert (w2.ema_loss[:-1] == 1).all()
