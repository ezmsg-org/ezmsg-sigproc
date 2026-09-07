"""Deprecation of the per-processor ``axis`` setting.

The mechanism lives in :mod:`ezmsg.baseproc.util.deprecation`; what belongs here
is the policy -- which release drops these settings, and the fact that it is
*this* distribution making the promise.

A processor that carries state *between* messages -- filter initial conditions,
a running mean, a sample buffer, a previous-sample cache -- can only do so along
the dimension messages accumulate along. Carrying it along a static axis is not
a smaller error but a different operation: that axis has the same length every
message, so the carried state applies message N's tail to message N+1's head at
the same coordinate, forever.

Which dimension that is belongs to the producer, and
:attr:`~ezmsg.util.messages.axisarray.AxisArray.chunk_dim` is where it says so.
A setting that lets a consumer disagree can only be used to be wrong, so it is
going away; see :func:`~ezmsg.baseproc.resolve_chunk_dim`.

During the deprecation window the setting is still honoured, so nothing changes
behaviour until it is removed. Two warnings partition the call sites:

* This module's construction-time :class:`FutureWarning` fires for *every* use,
  including a harmless ``axis="time"`` on a raw stream. It means "delete this".
* :func:`~ezmsg.baseproc.resolve_configured_chunk_dim`'s runtime warning fires
  only when the configured axis disagrees with a *declared* ``chunk_dim``. It
  means "deleting this will change what this stage computes".

To find every remaining call site in a pipeline, run its tests with
``-W error::FutureWarning``.
"""

import typing

from ezmsg.baseproc import suppress_axis_deprecation as suppress_axis_deprecation
from ezmsg.baseproc import warn_axis_deprecated as _warn_axis_deprecated

__all__ = [
    "AXIS_REMOVAL_VERSION",
    "suppress_axis_deprecation",
    "warn_axis_deprecated",
]

AXIS_REMOVAL_VERSION = "4.0"
"""Release that drops the deprecated ``axis`` settings. Deprecated in 3.8."""

_PACKAGE = "ezmsg-sigproc"


def warn_axis_deprecated(settings: typing.Any, field: str = "axis") -> None:
    """Warn that *settings*' ``field`` is deprecated, if it was actually set.

    Thin wrapper over :func:`ezmsg.baseproc.warn_axis_deprecated` that names this
    distribution and its removal release, so the call sites stay a single line
    and every message agrees about when the setting goes away.
    """
    _warn_axis_deprecated(settings, field, package=_PACKAGE, removal=AXIS_REMOVAL_VERSION)
