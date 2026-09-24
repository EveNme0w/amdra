import pytest

from amdra.evals.runner import select_cases


def test_limit_samples_across_scenarios_tagged_first(cases):
    picked = select_cases(cases, limit=4)
    assert len(picked) == 4
    assert len({c.scenario for c in picked}) == 4
    assert all(c.tags for c in picked)  # injection + policy_version scenarios come first


def test_limit_equal_to_scenario_count_covers_all(cases):
    n_scenarios = len({c.scenario for c in cases})
    picked = select_cases(cases, limit=n_scenarios)
    assert len({c.scenario for c in picked}) == n_scenarios


def test_filters(cases):
    assert {c.scenario for c in select_cases(cases, tags=["injection"])} == {
        "injection_receipt", "injection_narrative", "injection_narrative_obfuscated",
        "injection_narrative_homoglyph", "injection_narrative_multilingual",
        "injection_receipt_image_only"}
    only = select_cases(cases, scenarios=["amount_matches"])
    assert len(only) == 3 and {c.scenario for c in only} == {"amount_matches"}
    with pytest.raises(ValueError):
        select_cases(cases, scenarios=["nope"])


def test_sampling_is_deterministic(cases):
    ids = lambda cs: [c.dispute.dispute_id for c in cs]  # noqa: E731
    assert ids(select_cases(cases, limit=9)) == ids(select_cases(cases, limit=9))
