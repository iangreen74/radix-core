"""
Dataset Preparation for PEFT/LoRA Fine-tuning

This module provides reproducible dataset preparation using small, manageable
datasets suitable for research and testing.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import json

try:
    from datasets import Dataset, DatasetDict, load_dataset
    from transformers import AutoTokenizer
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    Dataset = DatasetDict = load_dataset = AutoTokenizer = None

from ...engine.radix_core.config_v2 import get_config
from ...engine.radix_core.logging import get_logger
from ...engine.radix_core.utils.timers import time_operation
from ...engine.radix_core.dryrun import DryRunGuard

logger = get_logger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for dataset preparation."""

    dataset_name: str = "ag_news"
    subset: Optional[str] = None
    split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15)  # train, val, test
    max_samples_per_split: int = 1000  # Keep datasets small for research
    max_sequence_length: int = 512
    text_column: str = "text"
    label_column: str = "label"
    cache_dir: str = "./data/cache"
    output_dir: str = "./data/processed"

    def __post_init__(self):
        """Validate configuration."""
        if sum(self.split_ratio) != 1.0:
            raise ValueError("Split ratios must sum to 1.0")

        if self.max_samples_per_split < 10:
            raise ValueError("max_samples_per_split must be at least 10")

        # Ensure directories are safe (local only)
        for path_str in [self.cache_dir, self.output_dir]:
            path = Path(path_str)
            if path.is_absolute() and not str(path).startswith(str(Path.home())):
                if not str(path).startswith(("/tmp", "/var/tmp", "./", "../")):
                    raise ValueError(f"Path {path_str} must be relative or in user directory")


class DatasetPreparator:
    """Prepares datasets for PEFT/LoRA fine-tuning with safety guards."""

    def __init__(self, config: DatasetConfig, tokenizer_name: str = None):
        """
        Initialize dataset preparator.

        Args:
            config: Dataset configuration
            tokenizer_name: Tokenizer to use (from global config if None)
        """
        if not DATASETS_AVAILABLE:
            raise ImportError("datasets library not available. Install with: pip install datasets")

        self.config = config
        self.global_config = get_config()
        self.tokenizer_name = tokenizer_name or self.global_config.ml.default_model_name

        # Initialize tokenizer
        self.tokenizer = None
        self.dataset_info: Dict[str, Any] = {}

        logger.info("Dataset preparator initialized",
                   dataset_name=config.dataset_name,
                   tokenizer_name=self.tokenizer_name,
                   max_samples=config.max_samples_per_split)

    def _load_tokenizer(self):
        """Load tokenizer for text processing."""
        if self.tokenizer is not None:
            return

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name,
                trust_remote_code=False,
                local_files_only=False,
                cache_dir=self.config.cache_dir
            )

            # Add pad token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.info("Tokenizer loaded successfully",
                       tokenizer_name=self.tokenizer_name,
                       vocab_size=self.tokenizer.vocab_size)

        except Exception as e:
            logger.error("Failed to load tokenizer", error=str(e))
            raise

    @DryRunGuard.protect
    def prepare_dataset(self) -> DatasetDict:
        """
        Prepare dataset for fine-tuning.

        Returns:
            Prepared dataset with train/validation/test splits
        """
        with time_operation(f"dataset_preparation_{self.config.dataset_name}"):
            logger.info("Starting dataset preparation",
                       dataset_name=self.config.dataset_name,
                       max_samples=self.config.max_samples_per_split)

            # Load raw dataset
            raw_dataset = self._load_raw_dataset()

            # Create splits
            dataset_splits = self._create_splits(raw_dataset)

            # Load tokenizer
            self._load_tokenizer()

            # Process and tokenize
            processed_dataset = self._process_and_tokenize(dataset_splits)

            # Save processed dataset
            self._save_processed_dataset(processed_dataset)

            # Update dataset info
            self._update_dataset_info(processed_dataset)

            logger.info("Dataset preparation completed",
                       train_samples=len(processed_dataset["train"]),
                       val_samples=len(processed_dataset["validation"]),
                       test_samples=len(processed_dataset["test"]))

            return processed_dataset

    def _load_raw_dataset(self) -> Dataset:
        """Load raw dataset from HuggingFace datasets."""
        try:
            logger.info("Loading raw dataset", dataset_name=self.config.dataset_name)

            # Load dataset with safety constraints
            if self.config.dataset_name == "ag_news":
                # AG News classification dataset (small and safe)
                dataset = load_dataset(
                    "ag_news",
                    cache_dir=self.config.cache_dir,
                    trust_remote_code=False
                )
                # Combine train and test for our own splitting
                combined = Dataset.from_dict({
                    "text": dataset["train"]["text"] + dataset["test"]["text"],
                    "label": dataset["train"]["label"] + dataset["test"]["label"]
                })

            elif self.config.dataset_name == "imdb":
                # IMDB sentiment dataset (small subset)
                dataset = load_dataset(
                    "imdb",
                    cache_dir=self.config.cache_dir,
                    trust_remote_code=False
                )
                # Take small subset
                combined = dataset["train"].select(range(min(5000, len(dataset["train"]))))

            else:
                # Generic dataset loading
                dataset = load_dataset(
                    self.config.dataset_name,
                    self.config.subset,
                    cache_dir=self.config.cache_dir,
                    trust_remote_code=False
                )

                # Use first available split
                split_name = list(dataset.keys())[0]
                combined = dataset[split_name]

            logger.info("Raw dataset loaded",
                       total_samples=len(combined),
                       columns=combined.column_names)

            return combined

        except Exception as e:
            logger.error("Failed to load raw dataset", error=str(e))
            raise

    def _create_splits(self, dataset: Dataset) -> DatasetDict:
        """Create train/validation/test splits."""
        total_samples = len(dataset)
        max_total = self.config.max_samples_per_split * 3  # For all splits

        if total_samples > max_total:
            # Sample subset to keep dataset manageable
            dataset = dataset.select(range(max_total))
            total_samples = max_total
            logger.info("Dataset subsampled for manageability",
                       original_size=len(dataset),
                       subsampled_size=total_samples)

        # Calculate split sizes
        train_size = int(total_samples * self.config.split_ratio[0])
        val_size = int(total_samples * self.config.split_ratio[1])
        test_size = total_samples - train_size - val_size

        # Create splits
        train_dataset = dataset.select(range(train_size))
        val_dataset = dataset.select(range(train_size, train_size + val_size))
        test_dataset = dataset.select(range(train_size + val_size, total_samples))

        splits = DatasetDict({
            "train": train_dataset,
            "validation": val_dataset,
            "test": test_dataset
        })

        logger.info("Dataset splits created",
                   train_size=len(train_dataset),
                   val_size=len(val_dataset),
                   test_size=len(test_dataset))

        return splits

    def _process_and_tokenize(self, dataset_splits: DatasetDict) -> DatasetDict:
        """Process and tokenize dataset splits."""
        def tokenize_function(examples):
            """Tokenize text examples."""
            # Handle different column names
            if self.config.text_column in examples:
                texts = examples[self.config.text_column]
            elif "text" in examples:
                texts = examples["text"]
            else:
                # Try to find text column
                text_columns = [col for col in examples.keys()
                              if any(keyword in col.lower()
                                   for keyword in ["text", "sentence", "content"])]
                if text_columns:
                    texts = examples[text_columns[0]]
                else:
                    raise ValueError("No text column found in dataset")

            # Tokenize
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=self.config.max_sequence_length,
                return_tensors=None  # Return lists, not tensors
            )

            return tokenized

        # Apply tokenization to all splits
        processed_splits = {}

        for split_name, split_dataset in dataset_splits.items():
            logger.info("Tokenizing split", split=split_name, size=len(split_dataset))

            tokenized_split = split_dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=split_dataset.column_names,  # Remove original columns
                desc=f"Tokenizing {split_name}"
            )

            # Add labels back if they exist
            if self.config.label_column in split_dataset.column_names:
                tokenized_split = tokenized_split.add_column(
                    "labels",
                    split_dataset[self.config.label_column]
                )

            processed_splits[split_name] = tokenized_split

        return DatasetDict(processed_splits)

    def _save_processed_dataset(self, dataset: DatasetDict):
        """Save processed dataset to disk."""
        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        dataset_path = output_path / f"{self.config.dataset_name}_processed"

        try:
            dataset.save_to_disk(str(dataset_path))
            logger.info("Processed dataset saved", path=str(dataset_path))

            # Save configuration
            config_path = dataset_path / "dataset_config.json"
            with open(config_path, 'w') as f:
                json.dump({
                    "dataset_name": self.config.dataset_name,
                    "tokenizer_name": self.tokenizer_name,
                    "max_sequence_length": self.config.max_sequence_length,
                    "split_ratio": self.config.split_ratio,
                    "max_samples_per_split": self.config.max_samples_per_split,
                    "created_at": "2024-09-28T10:39:28-07:00"
                }, f, indent=2)

        except Exception as e:
            logger.warning("Failed to save processed dataset", error=str(e))

    def _update_dataset_info(self, dataset: DatasetDict):
        """Update dataset information for tracking."""
        self.dataset_info = {
            "dataset_name": self.config.dataset_name,
            "tokenizer_name": self.tokenizer_name,
            "splits": {
                split_name: {
                    "size": len(split_data),
                    "columns": split_data.column_names
                }
                for split_name, split_data in dataset.items()
            },
            "config": {
                "max_sequence_length": self.config.max_sequence_length,
                "split_ratio": self.config.split_ratio,
                "max_samples_per_split": self.config.max_samples_per_split
            }
        }

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the prepared dataset."""
        return self.dataset_info.copy()


# Convenience functions
def prepare_dataset(dataset_name: str = "ag_news",
                   tokenizer_name: str = None,
                   max_samples: int = 1000) -> DatasetDict:
    """
    Convenience function to prepare a dataset.

    Args:
        dataset_name: Name of dataset to prepare
        tokenizer_name: Tokenizer to use
        max_samples: Maximum samples per split

    Returns:
        Prepared dataset
    """
    config = DatasetConfig(
        dataset_name=dataset_name,
        max_samples_per_split=max_samples
    )

    preparator = DatasetPreparator(config, tokenizer_name)
    return preparator.prepare_dataset()


def load_prepared_dataset(dataset_name: str,
                         data_dir: str = "./data/processed") -> Optional[DatasetDict]:
    """
    Load a previously prepared dataset.

    Args:
        dataset_name: Name of dataset to load
        data_dir: Directory containing processed datasets

    Returns:
        Loaded dataset or None if not found
    """
    if not DATASETS_AVAILABLE:
        logger.error("datasets library not available")
        return None

    dataset_path = Path(data_dir) / f"{dataset_name}_processed"

    if not dataset_path.exists():
        logger.warning("Processed dataset not found", path=str(dataset_path))
        return None

    try:
        dataset = DatasetDict.load_from_disk(str(dataset_path))
        logger.info("Processed dataset loaded", path=str(dataset_path))
        return dataset

    except Exception as e:
        logger.error("Failed to load processed dataset", error=str(e))
        return None


def get_sample_data() -> List[Dict[str, Any]]:
    """Get sample data for testing when datasets library is not available."""
    return [
        {"text": "The stock market rose today on positive earnings reports.", "label": 0},
        {"text": "Scientists discover new species in deep ocean exploration.", "label": 1},
        {"text": "Local team wins championship in thrilling overtime game.", "label": 2},
        {"text": "New smartphone features advanced AI capabilities.", "label": 3},
        {"text": "Climate change impacts discussed at international summit.", "label": 1},
        {"text": "Company announces record quarterly profits.", "label": 0},
        {"text": "Archaeological dig uncovers ancient artifacts.", "label": 1},
        {"text": "Basketball playoffs begin with exciting matchups.", "label": 2},
        {"text": "Latest software update improves security features.", "label": 3},
        {"text": "Renewable energy adoption reaches new milestone.", "label": 1}
    ]
