### Protein Stability Environment

The environment is designed to provide a set of tools for mutating a protein sequence to improve protein stability. It includes the following tools:

1. Describes bond types between any residues
2. Distance between two residues
3. Sequence based properties
4. Properties of a specific amino acid
5. Protein language model tools
6. Secondary structure annotation

The reward function is based on the ddG score between the original protein and the mutated protein.

## Dataset

Dataset used to evaluate the environment is based on the Megascale Stability Dataset. A pre-processed version of the source dataset is provided in the `aviary-storage` bucket under `baseline-envs/proteincrow-stability/v1`. It will be automatically downloaded when instantiating `ProteinStabilityDataset`.
