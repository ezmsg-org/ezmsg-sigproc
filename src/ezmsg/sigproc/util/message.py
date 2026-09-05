"""Message (AxisArray) utilities.

Also re-exports sample-message symbols from ezmsg.baseproc.util.message for
backwards compatibility; new code should import those directly from
ezmsg.baseproc instead.
"""

import typing

import ezmsg.core as ez
from ezmsg.baseproc.util.message import (
    SampleMessage,
    SampleTriggerMessage,
    is_sample_message,
)
from ezmsg.util.messages.axisarray import AxisArray

__all__ = [
    "SampleMessage",
    "SampleTriggerMessage",
    "has_samples_along",
    "is_empty_along",
    "is_sample_message",
    "resolve_chunk_dim",
    "resolve_configured_chunk_dim",
    "resolve_feature_dim",
    "resolve_transform_dim",
    "with_fingerprint",
]

STREAMING_DIMS: tuple[str, ...] = ("time",)
"""Default fallback chunk dimension, matching ``BaseStatefulTransformer``."""


def resolve_chunk_dim(message: AxisArray, streaming_dims: typing.Iterable[str] = STREAMING_DIMS) -> str:
    """The dimension successive messages accumulate along.

    This is the axis a processor that carries state *between* messages must
    operate on -- filter initial conditions, a running mean, a sample buffer,
    a previous-sample cache. Carrying such state along any other dimension is
    not a smaller error but a different operation: a static axis has the same
    length every message, so state carried across it applies message N's tail
    to message N+1's head at the same coordinate, forever.

    The producer renamed the dims and so is the only party that reliably knows
    which one grows; ``message.chunk_dim`` is that declaration. When a producer
    is silent, *streaming_dims* supplies the guess -- ``("time",)`` is right for
    a raw signal and wrong downstream of a windowing stage, where the message is
    ``(win, time, ch)`` and ``win`` is what grows.

    ``dims[0]`` is the last resort only. It is a position, not a meaning, and it
    breaks under :meth:`~ezmsg.util.messages.axisarray.AxisArray.transpose`.
    """
    if message.chunk_dim is not None:
        return message.chunk_dim
    for name in streaming_dims:
        if name in message.dims:
            return name
    return message.dims[0]


def resolve_configured_chunk_dim(
    processor: typing.Any,
    message: AxisArray,
    configured: str | None,
    legacy_default: str | None = None,
) -> str:
    """Resolve a state-carrying processor's axis, honouring an explicit setting.

    *configured* wins when set -- an explicit axis is an instruction, and
    removing that escape hatch would break every pipeline that passes the
    common ``axis="time"``. But when the producer *declared* a different chunk
    dimension, that disagreement is worth surfacing exactly once: the
    processor's cross-message state is about to be carried along an axis whose
    length is fixed, which is a different operation from the one the caller
    almost certainly meant.

    The warning fires only against a declared ``chunk_dim``, never against the
    :attr:`STREAMING_DIMS` guess -- warning on a guess would fire on every
    correctly-configured windowed pipeline whose producer is merely silent.

    :param legacy_default: The dimension this processor's ``axis`` setting used
        to default to, for the stages whose default was a hardcoded ``"time"``
        rather than a positional guess. Flipping those to follow ``chunk_dim``
        changes results wherever the chunk dimension is not ``"time"`` -- most
        obviously downstream of a windowing stage, where it is ``"win"`` -- and
        unlike an explicitly configured axis there is nothing in the settings to
        warn about. Passing the old default here surfaces exactly that
        population, once, and is dropped when the setting is removed.
    """
    resolved = resolve_chunk_dim(message, getattr(processor, "STREAMING_DIMS", STREAMING_DIMS))
    if configured is None:
        if (
            legacy_default is not None
            and resolved != legacy_default
            and legacy_default in message.dims
            and not getattr(processor, "_legacy_axis_default_warned", False)
        ):
            processor._legacy_axis_default_warned = True
            ez.logger.warning(
                f"{type(processor).__name__} used to operate on axis={legacy_default!r} by default; it now "
                f"follows the stream's chunk_dim={resolved!r}. This changes its output. The old behaviour was "
                f"carrying state across messages along {legacy_default!r}, whose length does not grow, so this "
                f"is a fix -- but pass axis={legacy_default!r} explicitly to keep the previous behaviour."
            )
        return resolved
    if (
        message.chunk_dim is not None
        and configured != message.chunk_dim
        and configured in message.dims
        and not getattr(processor, "_chunk_dim_mismatch_warned", False)
    ):
        processor._chunk_dim_mismatch_warned = True
        ez.logger.warning(
            f"{type(processor).__name__} is configured with axis={configured!r} but messages declare "
            f"chunk_dim={message.chunk_dim!r}. State carried between messages will be applied along "
            f"{configured!r}, whose length does not grow. Set axis=None to follow the declared chunk dimension."
        )
    return configured


def resolve_feature_dim(message: AxisArray, position: int = -1) -> str:
    """The dimension at *position*, skipping the chunk dimension.

    For processors whose axis is a *static* one -- channels, coordinate
    components, feature labels. ``chunk_dim`` is emphatically not the answer
    here, but the naive ``dims[position]`` can silently *be* the chunk
    dimension: a ``(ch, time)`` stream makes ``dims[-1]`` the accumulating axis,
    and an affine transform would then matmul across time while a slicer would
    discard samples.

    Falls back to ``dims[position]`` when the chunk dimension is all there is,
    which keeps 1-D messages working rather than raising on them.
    """
    candidates = [d for d in message.dims if d != message.chunk_dim]
    if not candidates:
        return message.dims[position]
    return candidates[position]


def resolve_transform_dim(message: AxisArray, streaming_dims: typing.Iterable[str] = STREAMING_DIMS) -> str:
    """The regularly-sampled dimension a transform consumes.

    Neither :func:`resolve_chunk_dim` nor :func:`resolve_feature_dim` fits a
    stage like :obj:`~ezmsg.sigproc.spectrum.Spectrum`, which needs the axis
    whose ``gain`` is a sample period and whose extent is the transform length:

    * On a raw ``(time, ch)`` stream that *is* the chunk dimension.
    * On windowed ``(win, time, ch)`` it is ``time`` -- ``win`` is what
      accumulates, but each window's spectrum is taken over ``time``.

    So: prefer the innermost non-chunk dimension carrying a ``LinearAxis``, and
    fall back to the chunk dimension when there is none. ``ch`` carries a
    ``CoordinateAxis`` (or no axis at all), so the raw case falls through
    correctly rather than transforming across channels.
    """
    chunk_dim = resolve_chunk_dim(message, streaming_dims)
    for name in reversed(message.dims):
        if name == chunk_dim:
            continue
        if isinstance(message.axes.get(name), AxisArray.LinearAxis):
            return name
    return chunk_dim


def with_fingerprint(axis: AxisArray.CoordinateAxis) -> AxisArray.CoordinateAxis:
    """Compute *axis*'s fingerprint now, and return the axis.

    Every stateful consumer reads the fingerprint of the coordinate axes that
    describe a stream's configuration, and the value is cached on the instance
    and pickled with it. Computing it at the point of construction therefore
    pays the checksum once, for everybody:

    * In this process, the axis object is reused for the life of the stream, so
      one call covers every message and every consumer downstream of it.
    * Across a process boundary it is better than that. Unpickling hands out a
      *new* axis object per message, so a cold axis is re-checksummed by the
      first consumer in every receiving process, on every message, forever.
      A primed one arrives with the answer already attached.

    Apply it to axes that describe the stream -- channel labels, frequency
    labels, feature labels -- not to per-message coordinates along the chunk
    dimension, whose fingerprint no consumer reads and whose data is new every
    message anyway.
    """
    axis.fingerprint  # noqa: B018 -- evaluated for the caching side effect
    return axis


def is_empty_along(message: AxisArray, dims: typing.Iterable[str]) -> bool:
    """True iff any of the named dims is present in ``message`` with zero length.

    Publish gates use this instead of ``data.size == 0`` so a message that is
    empty only along *other* axes — e.g. an upstream selection removed every
    channel while time samples remain — still flows downstream, preserving the
    stream's cadence for consumers that align or merge multiple sources.
    Dims not present in the message are ignored.
    """
    return any(d in message.dims and message.data.shape[message.get_axis_idx(d)] == 0 for d in dims)


def has_samples_along(message: AxisArray, dim: str) -> bool:
    """True iff ``dim`` is present in ``message`` with nonzero length.

    Stricter than ``not is_empty_along(...)``: the dim must exist. Drain loops
    use this to decide whether a chunk is real output, so that a placeholder
    lacking the axis entirely (e.g. ResampleProcessor's pre-init null template,
    ``dims=[""]``) counts as "nothing ready" rather than a publishable chunk.
    """
    return dim in message.dims and message.data.shape[message.get_axis_idx(dim)] > 0
