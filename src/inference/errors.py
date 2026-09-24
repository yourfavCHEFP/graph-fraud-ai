"""Errors raised while preparing inference artifacts."""


class InferenceConfigurationError(RuntimeError):
    """Base class for startup failures caused by invalid deployment inputs."""


class ArtifactMissingError(InferenceConfigurationError):
    """A required model artifact is missing."""


class LFSPointerError(InferenceConfigurationError):
    """A required artifact is an unmaterialized Git LFS pointer."""


class RegistryContractError(InferenceConfigurationError):
    """The model registry does not match its required schema."""


class CheckpointContractError(InferenceConfigurationError):
    """The checkpoint does not match the active feature/model contract."""
