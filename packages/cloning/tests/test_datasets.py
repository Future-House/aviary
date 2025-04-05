from cloning.dataset import CloningDataset


def test_datasets_reasonable() -> None:
    dataset = CloningDataset(split="all")
    all_seqqa = 15 * 50
    public_seqqa = all_seqqa * 0.8
    rna_questions = 50 * 0.8
    assert len(dataset.dataset) == public_seqqa - rna_questions

    dataset = CloningDataset(split="train")
    assert len(dataset.dataset) == (public_seqqa - rna_questions) * 0.9

    dataset = CloningDataset(subset_name="CloningScenarios")
    _ = dataset.get_new_env()

    dataset = CloningDataset(
        split="all", include_cannot_answer=True, include_convention=True
    )
    assert "Note:" in dataset.dataset.question[0]


def test_datasets_deterministic() -> None:
    cloning_dataset = CloningDataset(split="train")
    dataset = cloning_dataset.dataset
    print(dataset.id.values[0], dataset.id.values[-1])
    assert (
        dataset.id.values[0]
        == cloning_dataset.get_new_env_by_idx(0).problem_id
        == "00a4c679-8038-4898-beed-8717ff5c8874"
    )
    assert (
        dataset.id.values[-1]
        == cloning_dataset.get_new_env_by_idx(len(cloning_dataset) - 1).problem_id
        == "e59f2058-c73b-47c2-8777-6b7df3a95802"
    )

    dataset = CloningDataset(split="val").dataset
    print(dataset.id.values[0], dataset.id.values[-1])

    assert dataset.id.values[0] == "e5b8787e-e272-426b-969d-44b963d78d2c"
    assert dataset.id.values[-1] == "fff62bd8-2820-4e40-8bc7-0e62cde13c16"

    dataset = CloningDataset(split="all").dataset
    print(dataset.id.values[0], dataset.id.values[-1])

    assert dataset.id.values[0] == "00a4c679-8038-4898-beed-8717ff5c8874"
    assert dataset.id.values[-1] == "fff62bd8-2820-4e40-8bc7-0e62cde13c16"
