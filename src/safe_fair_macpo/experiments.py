from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentGroup:
    code: str
    description: str
    tasks: str
    methods: str
    seeds: str


PAPER_EXPERIMENTS = [
    ExperimentGroup(
        code="E1",
        description="Main safe-fair comparison",
        tasks="HalfCheetah-2x3-Fair, Ant-2x4-Fair, MultiGoal-Fair, ChargeShare-Fair",
        methods="MAPPO, MACPO, MAPPO-Lagrangian, DFRL/SOTO-GGF, reward-shaped variants, Safe-Fair MACPO",
        seeds="3",
    ),
    ExperimentGroup(
        code="E2",
        description="Safety-fairness trade-off sweep",
        tasks="Ant-2x4-Fair, ChargeShare-Fair",
        methods="Safe-Fair MACPO over fairness-budget sweep",
        seeds="3",
    ),
    ExperimentGroup(
        code="E3",
        description="Fairness-debt ablation",
        tasks="Ant-2x4-Fair, ChargeShare-Fair",
        methods="Safe-Fair MACPO, no debt, no envy, no recovery, reward-shaped fairness",
        seeds="5",
    ),
    ExperimentGroup(
        code="E4",
        description="Safety-first recovery ablation",
        tasks="Ant-2x4-Fair, ChargeShare-Fair",
        methods="Safe-Fair MACPO vs no recovery",
        seeds="5",
    ),
    ExperimentGroup(
        code="E5",
        description="Heterogeneity test",
        tasks="ChargeShare-Fair, Ant-2x4-Fair",
        methods="Safe-Fair MACPO with non-uniform target shares",
        seeds="3",
    ),
    ExperimentGroup(
        code="E6",
        description="Scalability test",
        tasks="ManyAgent-Ant-Fair",
        methods="Safe-Fair MACPO, local k-hop fairness, MACPO",
        seeds="3",
    ),
    ExperimentGroup(
        code="E7",
        description="Distribution-shift stress test",
        tasks="MultiGoal-Fair, Ant-2x4-Fair",
        methods="MACPO, reward-shaped MACPO-Fair, Safe-Fair MACPO",
        seeds="3",
    ),
]


def format_experiment_matrix() -> str:
    """Return the paper experiment matrix as a Markdown table."""
    lines = [
        "| Group | Description | Tasks | Methods | Seeds |",
        "| --- | --- | --- | --- | --- |",
    ]
    for exp in PAPER_EXPERIMENTS:
        lines.append(
            f"| {exp.code} | {exp.description} | {exp.tasks} | {exp.methods} | {exp.seeds} |"
        )
    return "\n".join(lines)
