"""GPU scheduler implementations."""

from .base import Scheduler, Job, GPU, Assignment, JobStatus
from .fifo import FIFOScheduler
from .srpt import SRPTScheduler

# Import all schedulers
from .drf import DRFScheduler
from .easy import EASYScheduler
from .heft import HEFTScheduler
from .bfd import BFDScheduler
from .gavel import GavelScheduler
from .radix_info_theory import RadixInfoTheoryScheduler
from .radix_softmax import RadixSoftmaxScheduler

# Scheduler registry
SCHEDULERS = {
    "fifo": FIFOScheduler,
    "srpt": SRPTScheduler,
    "drf": DRFScheduler,
    "easy": EASYScheduler,
    "heft": HEFTScheduler,
    "bfd": BFDScheduler,
    "gavel": GavelScheduler,
    "radix": RadixInfoTheoryScheduler,
    "radix_softmax": RadixSoftmaxScheduler,
}

__all__ = [
    "Scheduler", "Job", "GPU", "Assignment", "JobStatus",
    "FIFOScheduler", "SRPTScheduler", "SCHEDULERS"
]
