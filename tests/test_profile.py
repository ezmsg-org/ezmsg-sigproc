"""
Tests that profile utilities are correctly re-exported from ezmsg.baseproc.

Comprehensive tests for the profile module are in ezmsg-baseproc.
"""

from ezmsg.sigproc.util.profile import (
    HEADER,
    _setup_logger,
    get_logger_path,
    profile_method,
    profile_subpub,
)


def test_reexports():
    """Verify all profile utilities are re-exported from ezmsg.baseproc."""
    # These imports should work and reference the same objects as ezmsg.baseproc
    from ezmsg.baseproc.util.profile import (
        HEADER as BASEPROC_HEADER,
    )
    from ezmsg.baseproc.util.profile import (
        _setup_logger as baseproc_setup_logger,
    )
    from ezmsg.baseproc.util.profile import (
        get_logger_path as baseproc_get_logger_path,
    )
    from ezmsg.baseproc.util.profile import (
        profile_method as baseproc_profile_method,
    )
    from ezmsg.baseproc.util.profile import (
        profile_subpub as baseproc_profile_subpub,
    )

    assert HEADER is BASEPROC_HEADER
    assert _setup_logger is baseproc_setup_logger
    assert get_logger_path is baseproc_get_logger_path
    assert profile_method is baseproc_profile_method
    assert profile_subpub is baseproc_profile_subpub
