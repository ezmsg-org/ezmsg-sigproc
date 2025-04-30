from types import NoneType
from typing import Any, Optional, Union

from ezmsg.sigproc.util.typeresolution import check_message_type_compatibility


# Simple mock message types for testing type compatibility
class MockMessageA:
    pass


class MockMessageB(MockMessageA):
    pass


class MockMessageC:
    pass


class TestCheckMessageTypeCompatibility:
    """
    Unit tests for the check_message_type_compatibility function.
    """

    def test_concrete_subclass_compatibility(self):
        assert check_message_type_compatibility(MockMessageB, MockMessageA)

        # Test incompatible types
        assert not check_message_type_compatibility(MockMessageC, MockMessageA)
        assert not check_message_type_compatibility(MockMessageA, MockMessageB)
        assert not check_message_type_compatibility(None, MockMessageA)
        assert not check_message_type_compatibility(MockMessageA, None)
        assert not check_message_type_compatibility(NoneType, MockMessageA)
        assert not check_message_type_compatibility(MockMessageA, NoneType)

    def test_none_and_any_compatibility(self):
        assert check_message_type_compatibility(None, None)
        assert check_message_type_compatibility(NoneType, None)
        assert check_message_type_compatibility(None, NoneType)
        assert check_message_type_compatibility(MockMessageA, Any)
        assert check_message_type_compatibility(Any, MockMessageA)
        assert check_message_type_compatibility(None, Any)
        assert check_message_type_compatibility(Any, None)

    def test_optional_compatibility(self):
        assert check_message_type_compatibility(MockMessageA, Optional[MockMessageA])
        assert check_message_type_compatibility(MockMessageB, Optional[MockMessageA])
        assert check_message_type_compatibility(None, Optional[MockMessageA])
        assert check_message_type_compatibility(NoneType, Optional[MockMessageA])
        assert check_message_type_compatibility(
            Optional[MockMessageA], Optional[MockMessageA]
        )
        assert not check_message_type_compatibility(Optional[MockMessageA], None)
        assert not check_message_type_compatibility(Optional[MockMessageA], NoneType)
        assert not check_message_type_compatibility(
            Optional[MockMessageA], MockMessageA
        )
        assert not check_message_type_compatibility(
            MockMessageA, Optional[MockMessageB]
        )
        assert not check_message_type_compatibility(
            MockMessageC, Optional[MockMessageA]
        )

    def test_union_compatibility(self):
        assert check_message_type_compatibility(MockMessageA, Union[MockMessageA, int])
        assert check_message_type_compatibility(MockMessageB, Union[MockMessageA, int])
        assert check_message_type_compatibility(None, Union[MockMessageA, None])
        assert check_message_type_compatibility(NoneType, Union[MockMessageA, None])
        assert check_message_type_compatibility(
            Union[MockMessageA, int], Union[MockMessageA, int]
        )
        assert check_message_type_compatibility(
            Union[MockMessageA, None], Optional[MockMessageA]
        )
        assert check_message_type_compatibility(
            Optional[MockMessageA], Union[MockMessageA, None]
        )
        assert check_message_type_compatibility(
            Union[MockMessageB, int], Union[MockMessageA, int, MockMessageC]
        )
        assert not check_message_type_compatibility(Union[MockMessageA, None], None)
        assert not check_message_type_compatibility(Union[MockMessageA, None], NoneType)
        assert not check_message_type_compatibility(
            Union[MockMessageA, int], MockMessageA
        )
        assert not check_message_type_compatibility(
            Union[MockMessageA, int, MockMessageC], Union[MockMessageA, int]
        )
        assert not check_message_type_compatibility(
            MockMessageC, Union[MockMessageA, int]
        )

    def test_union_operator_compatibility(self):
        assert check_message_type_compatibility(MockMessageA, MockMessageA | int)
        assert check_message_type_compatibility(MockMessageB, MockMessageA | int)
        assert check_message_type_compatibility(None, MockMessageA | None)
        assert check_message_type_compatibility(NoneType, MockMessageA | None)
        assert check_message_type_compatibility(MockMessageA | int, MockMessageA | int)
        assert check_message_type_compatibility(
            MockMessageB | bool, Union[MockMessageA, int]
        )
        assert check_message_type_compatibility(
            Union[MockMessageB, bool], MockMessageA | int
        )
        assert check_message_type_compatibility(
            MockMessageA | None, Optional[MockMessageA]
        )
        assert check_message_type_compatibility(
            Optional[MockMessageA], MockMessageA | None
        )
        assert check_message_type_compatibility(
            MockMessageB | int, MockMessageA | int | MockMessageC
        )
        assert not check_message_type_compatibility(MockMessageA | None, None)
        assert not check_message_type_compatibility(MockMessageA | None, NoneType)
        assert not check_message_type_compatibility(MockMessageA | int, MockMessageA)
        assert not check_message_type_compatibility(
            MockMessageA | int | MockMessageC, MockMessageA | int
        )
        assert not check_message_type_compatibility(MockMessageC, MockMessageA | int)
