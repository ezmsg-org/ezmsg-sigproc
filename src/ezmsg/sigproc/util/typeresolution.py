from types import UnionType
import typing
from typing_extensions import get_original_bases


def resolve_typevar(cls: type, target_typevar: typing.TypeVar) -> type:
    """
    Resolve the concrete type bound to a TypeVar in a class hierarchy.
    This function traverses the method resolution order (MRO) of the class
    and checks the original bases of each class in the MRO for the TypeVar.
    If the TypeVar is found, it returns the concrete type bound to it.
    If the TypeVar is not found, it raises a TypeError.
    Args:
        cls (type): The class to inspect.
        target_typevar (typing.TypeVar): The TypeVar to resolve.
    Returns:
        type: The concrete type bound to the TypeVar.
    """
    for base in cls.__mro__:
        orig_bases = get_original_bases(base)
        for orig_base in orig_bases:
            origin = typing.get_origin(orig_base)
            if origin is None:
                continue
            params = getattr(origin, "__parameters__", ())
            if not params:
                continue
            if target_typevar in params:
                index = params.index(target_typevar)
                args = typing.get_args(orig_base)
                try:
                    return args[index]
                except IndexError:
                    pass
    raise TypeError(f"Could not resolve {target_typevar} in {cls}")


TypeLike = typing.Union[type[typing.Any], typing.Any, type(None), None]


def check_message_type_compatibility(type1: TypeLike, type2: TypeLike) -> bool:
    """
    Check if two types are compatible for message passing.
    Returns True if:
    - Both are None/NoneType
    - Either is typing.Any
    - type1 is a subclass of type2, which includes
        - type1 and type2 are concrete types and type1 is a subclass of type2
        - type1 is None/NoneType and type2 is typing.Optional, or
        - type1 is subtype of the non-None inner type of type2 if type2 is Optional
    - type1 is a Union/Optional type and all inner types are compatible with type2
    Args:
        type1: First type to compare
        type2: Second type to compare
    Returns:
        bool: True if the types are compatible, False otherwise
    """
    # If either is Any, they are compatible
    if type1 is typing.Any or type2 is typing.Any:
        return True

    # Handle None as NoneType
    if type1 is None:
        type1 = type(None)
    if type2 is None:
        type2 = type(None)

    # Handle if type1 is Optional/Union type
    if typing.get_origin(type1) in {typing.Union, UnionType}:
        return all(
            check_message_type_compatibility(inner_type, type2)
            for inner_type in typing.get_args(type1)
        )

    # Regular issubclass check. Handles cases like:
    # - type1 is a subclass of concrete type2
    # - type1 is a subclass of the inner type of type2 if type2 is Optional
    # - type1 is a subclass of one of the inner types of type2 if type2 is Union
    # - type1 is NoneType and type2 is Optional or Union[None, ...] or Union[NoneType, ...]
    try:
        return issubclass(type1, type2)
    except TypeError:
        return False
