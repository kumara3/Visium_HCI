import numpy as np
import pandas as pd


def summarize_roi_for_rq3(selected_df: pd.DataFrame, roi_stats: dict) -> dict:
    """
    Build a compact evidence summary from the selected ROI.
    This is the grounding layer for the assistant.
    """
    if selected_df.empty:
        return {
            "roi_size": 0,
            "dominant_region": "unknown",
            "tumor_fraction_estimate": np.nan,
            "mean_tumor_probability": np.nan,
            "mean_tumor_gradient": np.nan,
            "mean_uncertainty": np.nan,
            "boundary_class_estimate": "unknown",
            "top_evidence_points": [],
        }

    region_counts = selected_df["region"].value_counts()
    dominant_region = region_counts.idxmax() if not region_counts.empty else "unknown"

    if "invasion_front" in region_counts and region_counts["invasion_front"] == region_counts.max():
        boundary_class_estimate = "invasion_front"
    elif "adjacent_normal" in region_counts and region_counts["adjacent_normal"] == region_counts.max():
        boundary_class_estimate = "adjacent_normal"
    else:
        boundary_class_estimate = dominant_region

    evidence_points = [
        f"ROI contains {len(selected_df)} selected spots.",
        f"Estimated tumor fraction is {roi_stats['tumor_fraction_estimate']:.2%}.",
        f"Mean tumor probability is {roi_stats['mean_tumor_probability']:.3f}.",
        f"Mean tumor gradient is {roi_stats['mean_tumor_gradient']:.3f}.",
        f"Dominant region label is {dominant_region}.",
        f"Boundary class estimate is {boundary_class_estimate}.",
    ]

    if not np.isnan(roi_stats["mean_uncertainty"]):
        evidence_points.append(
            f"Mean uncertainty is {roi_stats['mean_uncertainty']:.3f}."
        )

    return {
        "roi_size": len(selected_df),
        "dominant_region": dominant_region,
        "tumor_fraction_estimate": roi_stats["tumor_fraction_estimate"],
        "mean_tumor_probability": roi_stats["mean_tumor_probability"],
        "mean_tumor_gradient": roi_stats["mean_tumor_gradient"],
        "mean_uncertainty": roi_stats["mean_uncertainty"],
        "boundary_class_estimate": boundary_class_estimate,
        "top_evidence_points": evidence_points,
    }


def generate_grounded_assistant_output(evidence_summary: dict) -> dict:
    """
    Produce a grounded assistant response.
    Rule-based proxy for an LLM: every statement is tied to ROI evidence.
    """
    if evidence_summary["roi_size"] == 0:
        return {
            "assistant_hypothesis": "No ROI is selected, so I cannot form a grounded hypothesis yet.",
            "assistant_next_step": "Select an ROI and inspect its tumor fraction, boundary class, and uncertainty.",
            "assistant_evidence_used": [],
            "assistant_confidence_note": "No evidence available.",
        }

    tumor_fraction = evidence_summary["tumor_fraction_estimate"]
    gradient = evidence_summary["mean_tumor_gradient"]
    uncertainty = evidence_summary["mean_uncertainty"]
    boundary = evidence_summary["boundary_class_estimate"]
    dominant_region = evidence_summary["dominant_region"]

    if boundary == "invasion_front" or gradient >= 0.25:
        hypothesis = (
            "This ROI may correspond to an invasive front or transition region, "
            "because it shows substantial local change and a boundary-like regional pattern."
        )
        next_step = (
            "Compare this ROI with neighboring ROIs and inspect whether uncertainty remains high "
            "along the putative boundary."
        )
    elif tumor_fraction >= 0.6 and dominant_region == "tumor_core":
        hypothesis = (
            "This ROI appears tumor-dominant, suggesting a likely tumor core rather than a boundary region."
        )
        next_step = (
            "Compare this ROI against nearby lower-probability regions to identify where tumor intensity begins to drop."
        )
    elif boundary == "adjacent_normal":
        hypothesis = (
            "This ROI appears more consistent with adjacent normal tissue near a tumor boundary."
        )
        next_step = (
            "Inspect nearby higher-gradient regions to determine where the transition into invasive tissue begins."
        )
    else:
        hypothesis = (
            "This ROI appears mixed or weakly classified, so it may represent an ambiguous transition area."
        )
        next_step = (
            "Inspect neighboring ROIs and compare uncertainty, tumor fraction, and gradient before making a stronger claim."
        )

    if not np.isnan(uncertainty):
        if uncertainty >= 0.45:
            confidence_note = "Interpret this suggestion cautiously because the ROI has high uncertainty."
        elif uncertainty >= 0.25:
            confidence_note = "This suggestion has moderate support, but uncertainty should still be checked."
        else:
            confidence_note = "This suggestion is better supported because the ROI uncertainty is relatively low."
    else:
        confidence_note = "Uncertainty is unavailable for this ROI."

    return {
        "assistant_hypothesis": hypothesis,
        "assistant_next_step": next_step,
        "assistant_evidence_used": evidence_summary["top_evidence_points"],
        "assistant_confidence_note": confidence_note,
    }


def score_rq3_response(
    user_hypothesis: str,
    user_next_step: str,
    assistant_output: dict,
    evidence_checks: list,
    trust_rating: int,
    revised_after_assistant: str
) -> dict:
    """
    Lightweight proxy scoring for RQ3.
    Useful for structured logs; human coding can still be added later.
    """
    user_hypothesis = (user_hypothesis or "").strip()
    user_next_step = (user_next_step or "").strip()
    revised_after_assistant = (revised_after_assistant or "").strip()

    hypothesis_length = len(user_hypothesis.split())
    next_step_length = len(user_next_step.split())

    evidence_keywords = [
        "tumor fraction",
        "uncertainty",
        "gradient",
        "boundary",
        "region",
        "tumor probability",
        "invasion",
        "adjacent",
        "roi",
    ]
    combined_text = f"{user_hypothesis} {user_next_step}".lower()
    evidence_keyword_hits = sum(1 for kw in evidence_keywords if kw in combined_text)

    planning_keywords = [
        "compare",
        "inspect",
        "review",
        "check",
        "validate",
        "analyze",
        "neighbor",
        "uncertainty",
        "region",
        "boundary",
    ]
    planning_hits = sum(1 for kw in planning_keywords if kw in user_next_step.lower())

    checked_evidence_count = len(evidence_checks)
    trust_norm = (trust_rating - 1) / 6.0
    revised_word_count = len(revised_after_assistant.split())
    revised_from_assistant = int(revised_word_count > 0)

    return {
        "rq3_hypothesis_word_count": hypothesis_length,
        "rq3_next_step_word_count": next_step_length,
        "rq3_evidence_keyword_hits": evidence_keyword_hits,
        "rq3_planning_keyword_hits": planning_hits,
        "rq3_checked_evidence_count": checked_evidence_count,
        "rq3_trust_norm": trust_norm,
        "rq3_revised_from_assistant": revised_from_assistant,
        "rq3_revised_word_count": revised_word_count,
        "rq3_assistant_hypothesis": assistant_output["assistant_hypothesis"],
        "rq3_assistant_next_step": assistant_output["assistant_next_step"],
        "rq3_assistant_confidence_note": assistant_output["assistant_confidence_note"],
        "rq3_assistant_evidence_count": len(assistant_output["assistant_evidence_used"]),
    }