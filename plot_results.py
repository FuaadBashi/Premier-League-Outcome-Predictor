"""Draw docs/results.png from a fresh evaluation run.

The figure is regenerated from the model, never typed in, so it cannot drift from what
predict.py reports. Needs matplotlib, which the model itself does not:

    pip install matplotlib
    python plot_results.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from features import (  # noqa: E402
    BASE_PREDICTORS,
    RESULT_LABELS,
    ROLLING_PREDICTORS,
    add_base_features,
    add_rolling_features,
    load_matches,
)
from predict import DEFAULT_DATA, baseline_accuracy, evaluate  # noqa: E402

OUT = Path(__file__).parent / "docs/results.png"
SPLIT = pd.Timestamp("2023-08-10")

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
MODEL, BASELINE = "#2a78d6", "#a8a79f"
# One hue, light to dark, for magnitude.
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]


def main() -> None:
    matches = add_base_features(load_matches(DEFAULT_DATA), SPLIT)
    fixture = evaluate("Fixture only", matches, BASE_PREDICTORS, SPLIT)
    form = evaluate(
        "Fixture + recent form",
        add_rolling_features(matches),
        BASE_PREDICTORS + ROLLING_PREDICTORS,
        SPLIT,
    )
    baseline = baseline_accuracy(matches, SPLIT)

    order = ["W", "D", "L"]
    table = pd.crosstab(
        form.predictions["actual"].map(RESULT_LABELS),
        form.predictions["predicted"].map(RESULT_LABELS),
    ).reindex(index=order, columns=order, fill_value=0)

    plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": INK_2})
    fig, (bars, heat) = plt.subplots(
        1, 2, figsize=(10, 3.6), dpi=200, gridspec_kw={"width_ratios": [1.35, 1]}
    )
    fig.patch.set_facecolor(SURFACE)

    labels = ["Baseline: always predict\nthe most common result", fixture.name, form.name]
    values = [baseline, fixture.accuracy, form.accuracy]
    colors = [BASELINE, MODEL, MODEL]
    y = range(len(values))
    bars.barh(y, values, height=0.5, color=colors)
    for i, v in zip(y, values, strict=True):
        bars.text(v + 0.01, i, f"{v:.1%}", va="center", color=INK, fontweight="bold")
    bars.set_yticks(list(y), labels, color=INK)
    bars.set_xlim(0, 0.7)
    bars.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    bars.tick_params(colors=INK_2, length=0)
    bars.grid(axis="x", color=GRID, linewidth=0.8)
    bars.set_axisbelow(True)
    for side in ("top", "right", "left", "bottom"):
        bars.spines[side].set_visible(False)
    bars.set_facecolor(SURFACE)
    bars.set_title("Accuracy on the unseen 2023–24 season", loc="left", fontweight="bold")

    cmap = LinearSegmentedColormap.from_list("blue", RAMP)
    heat.imshow(table.values, cmap=cmap, vmin=0)
    peak = table.values.max()
    for r, actual in enumerate(order):
        for c, predicted in enumerate(order):
            n = int(table.loc[actual, predicted])
            heat.text(
                c,
                r,
                n,
                ha="center",
                va="center",
                fontweight="bold",
                color="white" if n > peak * 0.55 else INK,
            )
    names = {"W": "Win", "D": "Draw", "L": "Loss"}
    heat.set_xticks(range(3), [names[k] for k in order])
    heat.set_yticks(range(3), [names[k] for k in order])
    heat.set_xlabel("Predicted")
    heat.set_ylabel("Actual")
    heat.tick_params(colors=INK, length=0)
    for side in heat.spines.values():
        side.set_visible(False)
    heat.set_title("Where the predictions land", loc="left", fontweight="bold")

    draws = int(table.loc["D"].sum())
    fig.text(
        0.01,
        0.01,
        f"The model never predicts a draw, so all {draws} draws count as misses. "
        f"Test set: {form.test_rows} team-matches from {SPLIT.date()}.",
        color=INK_2,
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, facecolor=SURFACE)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
