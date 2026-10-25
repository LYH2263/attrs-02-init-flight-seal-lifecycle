# SPDX-License-Identifier: MIT

from __future__ import annotations


class FrozenError(AttributeError):
    """
    A frozen/immutable instance or attribute have been attempted to be
    modified.

    It mirrors the behavior of ``namedtuples`` by using the same error message
    and subclassing `AttributeError`.

    .. versionadded:: 20.1.0
    """

    def __init__(self):
        msg = "can't set attribute"
        super().__init__(msg)
        self.msg = msg


class FrozenInstanceError(FrozenError):
    """
    A frozen instance has been attempted to be modified.

    .. versionadded:: 16.1.0
    """


class FrozenAttributeError(FrozenError):
    """
    A frozen attribute has been attempted to be modified.

    .. versionadded:: 20.1.0
    """


class InitFlightError(AttributeError):
    """
    Base class for violations of the construction flight field-sealing
    lifecycle.

    While an *attrs*-generated initializer runs, the instance goes through
    the observable phases ``pre_init``, ``assigning``, ``validating``,
    ``post_init`` and finally ``ready``.  Hooks that receive the partially
    constructed instance (``__attrs_pre_init__``, ``Converter(takes_self)``,
    ``Factory(takes_self)`` and validators) must not write to attribute
    storage names outside of the current planned assignment -- neither via
    `object.__setattr__` nor via a generated ``__setattr__``.  Doing so
    raises one of the two subclasses; the construction is then in the
    ``aborted`` phase.

    Attributes:
        name (str): The attribute storage name (``Attribute.name``, not the
            ``__init__`` alias) that the hook attempted to write.
        phase (str): The construction flight phase in which the write was
            attempted.

    .. versionadded:: 26.2.0
    """

    def __init__(self, name, phase, msg):
        super().__init__(msg)
        self.name = name
        self.phase = phase
        self.msg = msg

    def __str__(self):
        return str(self.msg)


class SealedAttributeError(InitFlightError):
    """
    A construction hook attempted to write an attribute whose planned
    assignment has already completed in this flight.

    Such fields are *sealed* until ``__attrs_post_init__`` runs; reading
    them is always allowed, rewriting them (including through
    `object.__setattr__`) aborts the construction.

    .. versionadded:: 26.2.0
    """

    def __init__(self, name, phase):
        msg = (
            f"attribute {name!r} is already assigned and sealed during "
            f"construction (phase {phase!r}); it can only be written again "
            "from __attrs_post_init__"
        )
        super().__init__(name, str(phase), msg)


class UnassignedAttributeError(InitFlightError):
    """
    A construction hook attempted to write an attribute whose planned
    assignment has not happened yet in this flight (an *early write*), or
    that is never assigned by the generated initializer.

    Early writes are rejected so that they cannot silently become field
    initial values or be overwritten by the later planned assignment.

    .. versionadded:: 26.2.0
    """

    def __init__(self, name, phase):
        msg = (
            f"attribute {name!r} has not been assigned yet during "
            f"construction (phase {phase!r}); hooks must not pre-write "
            "fields before their planned assignment"
        )
        super().__init__(name, str(phase), msg)


class AttrsAttributeNotFoundError(ValueError):
    """
    An *attrs* function couldn't find an attribute that the user asked for.

    .. versionadded:: 16.2.0
    """


class NotAnAttrsClassError(ValueError):
    """
    A non-*attrs* class has been passed into an *attrs* function.

    .. versionadded:: 16.2.0
    """


class DefaultAlreadySetError(RuntimeError):
    """
    A default has been set when defining the field and is attempted to be reset
    using the decorator.

    .. versionadded:: 17.1.0
    """


class UnannotatedAttributeError(RuntimeError):
    """
    A class with ``auto_attribs=True`` has a field without a type annotation.

    .. versionadded:: 17.3.0
    """


class PythonTooOldError(RuntimeError):
    """
    It was attempted to use an *attrs* feature that requires a newer Python
    version.

    .. versionadded:: 18.2.0
    """


class NotCallableError(TypeError):
    """
    A field requiring a callable has been set with a value that is not
    callable.

    .. versionadded:: 19.2.0
    """

    def __init__(self, msg, value):
        super(TypeError, self).__init__(msg, value)
        self.msg = msg
        self.value = value

    def __str__(self):
        return str(self.msg)
