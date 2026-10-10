"""The model card documents every feature the contract emits, and nothing else."""

from iudbms.contract import FEATURE_NAMES


def test_every_contract_feature_has_a_unit_and_definition():
    from eval.model_card import FEATURE_DOC
    assert set(FEATURE_DOC) == set(FEATURE_NAMES), (
        f"undocumented: {sorted(set(FEATURE_NAMES) - set(FEATURE_DOC))}; "
        f"stale: {sorted(set(FEATURE_DOC) - set(FEATURE_NAMES))}")
    for name, (unit, definition) in FEATURE_DOC.items():
        assert unit and len(definition) > 10, name
