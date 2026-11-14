"""The promotion gate (Decision D9, tolerance approved in Milestone 5).

On the gate window, the challenger must
  1. beat the champion's overall WAPE, for pickups and for drop-offs;
  2. be no more than `tolerance` worse than the champion on any horizon or volume tier;
  3. beat each baseline's overall WAPE, unless the champion does not beat it either; then
     beating the champion is enough (owner decision, 2026-11-14).
Rule 3 keeps a model worse than a baseline from being promoted over one that is better,
without keeping a worse champion in place when both trail the baseline. Every comparison
is recorded, so a rejection says exactly which check failed.
"""
import math

from bikeshare.modeling.dataset import TARGETS

GROUPS = ("horizon", "tier")


def _check(name: str, target: str, group: str, label: str, challenger: float,
           reference: str, reference_value: float, limit: float, waived: bool = False) -> dict:
    """Beating a model or baseline must be strict; staying within tolerance may touch it."""
    if math.isnan(challenger) or math.isnan(limit):
        passed = True  # a slice with no demand has no WAPE and cannot fail the gate
    elif name == "within_tolerance":
        passed = challenger <= limit
    else:
        passed = challenger < limit
    return {"check": name, "target": target, "group": group, "label": label,
            "reference": reference, "challenger_wape": challenger,
            "reference_wape": reference_value, "limit": limit, "passed": passed,
            "waived": waived}


def compare(challenger: dict, champion: dict, baselines: dict[str, dict],
            tolerance: float) -> dict:
    """Each argument is an evaluate() result: {demand: breakdown, ..., "net_flow": {...}}."""
    checks = []
    for target in TARGETS:
        mine, theirs = challenger[target], champion[target]
        overall = mine["overall"]["wape"]
        checks.append(_check("beats_champion", target, "all", "all", overall, "champion",
                             theirs["overall"]["wape"], theirs["overall"]["wape"]))
        for group in GROUPS:
            for label, scores in theirs[group].items():
                value = mine[group].get(label, {}).get("wape", math.nan)
                checks.append(_check("within_tolerance", target, group, label, value,
                                     "champion", scores["wape"],
                                     scores["wape"] * (1 + tolerance)))
        champion_wape = theirs["overall"]["wape"]
        for name, scores in baselines.items():
            reference = scores[target]["overall"]["wape"]
            # The champion trails this baseline too: the challenger need only beat the champion.
            waived = champion_wape >= reference
            checks.append(_check("beats_baseline", target, "all", "all", overall, name,
                                 reference, champion_wape if waived else reference, waived))
    failed = [check for check in checks if not check["passed"]]
    return {"passed": not failed, "tolerance": tolerance, "checks": checks,
            "failed": [describe(check) for check in failed],
            "waived": [describe(check) for check in checks if check["waived"]]}


def describe(check: dict) -> str:
    where = "overall" if check["group"] == "all" else f"{check['group']} {check['label']}"
    verb = {"beats_champion": "does not beat", "beats_baseline": "does not beat",
            "within_tolerance": "is more than the tolerance worse than"}[check["check"]]
    if check["waived"]:
        return (f"{check['target']} {where}: {check['reference']} "
                f"({check['reference_wape']:.4f}) beats the champion too, so the challenger "
                f"only had to beat the champion ({check['limit']:.4f})")
    return (f"{check['target']} {where}: challenger WAPE {check['challenger_wape']:.4f} "
            f"{verb} {check['reference']} ({check['reference_wape']:.4f})")
