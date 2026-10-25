# SPDX-License-Identifier: MIT

"""
Tests for the construction flight: per-instance observable phases and the
field-sealing lifecycle of generated ``__init__`` / ``__attrs_init__``.
"""

import copy
import gc
import pickle
import threading

import pytest

import attr
import attrs

from attr import Converter, Factory, attrib
from attr._compat import PY_3_13_PLUS
from attr.exceptions import (
    FrozenInstanceError,
    InitFlightError,
    SealedAttributeError,
    UnassignedAttributeError,
)


P = attr.InitPhase


def decorate(cls, *, slots, frozen=False, **kw):
    return attr.attrs(cls, slots=slots, frozen=frozen, **kw)


# Module-level (picklable) classes for the setstate fork tests.
_pickle_convert_calls = []
_pickle_self_phases = []


def _pickle_plain_converter(v):
    _pickle_convert_calls.append(v)
    return v


def _pickle_self_converter(v, self):
    _pickle_convert_calls.append(v)
    _pickle_self_phases.append(attr.get_init_phase(self))
    return v


@attr.s(slots=True, frozen=True)
class _PickleSlotted:
    x = attrib(converter=_pickle_plain_converter)
    y = attrib(converter=Converter(_pickle_self_converter, takes_self=True))


@attr.s(getstate_setstate=True)
class _PickleDict:
    x = attrib(converter=_pickle_plain_converter)
    y = attrib(converter=Converter(_pickle_self_converter, takes_self=True))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class TestPublicAPI:
    def test_phases_and_values(self):
        assert [p.value for p in attr.InitPhase] == [
            "pre_init",
            "assigning",
            "validating",
            "post_init",
            "ready",
            "aborted",
        ]
        assert isinstance(attr.InitPhase.READY, str)
        assert attr.InitPhase("assigning") is attr.InitPhase.ASSIGNING

    def test_public_namespaces(self):
        assert attrs.InitPhase is attr.InitPhase
        assert attrs.get_init_phase is attr.get_init_phase
        assert attrs.get_init_field is attr.get_init_field

    def test_error_hierarchy_and_payload(self):
        assert issubclass(SealedAttributeError, InitFlightError)
        assert issubclass(UnassignedAttributeError, InitFlightError)
        assert issubclass(InitFlightError, AttributeError)

        err = SealedAttributeError("x", P.ASSIGNING)
        assert err.name == "x"
        assert err.phase == "assigning"
        assert "x" in str(err)

    def test_ready_is_the_default(self):
        @attr.s
        class C:
            x = attrib()

        assert attr.get_init_phase(C(1)) is P.READY
        assert attr.get_init_field(C(1)) is None

    def test_non_attrs_object_is_ready(self):
        class Plain:
            pass

        assert attr.get_init_phase(Plain()) is P.READY
        assert attr.get_init_field(Plain()) is None


# ---------------------------------------------------------------------------
# (1) Phase observability
# ---------------------------------------------------------------------------


class TestPhases:
    @pytest.mark.parametrize("slots", [True, False])
    @pytest.mark.parametrize("frozen", [True, False])
    def test_full_sequence(self, slots, frozen):
        log = []

        class C:
            a = attrib(
                converter=Converter(
                    lambda v, self: (
                        log.append(
                            (
                                attr.get_init_phase(self),
                                attr.get_init_field(self),
                            )
                        )
                        or v
                    ),
                    takes_self=True,
                )
            )
            b = attrib(
                default=Factory(
                    lambda self: (
                        log.append(
                            (
                                attr.get_init_phase(self),
                                attr.get_init_field(self),
                            )
                        )
                        or 7
                    ),
                    takes_self=True,
                )
            )

            def __attrs_pre_init__(self):
                log.append(
                    (attr.get_init_phase(self), attr.get_init_field(self))
                )

            def __attrs_post_init__(self):
                log.append(
                    (attr.get_init_phase(self), attr.get_init_field(self))
                )

        C = decorate(C, slots=slots, frozen=frozen)
        c = C(1)

        assert (c.a, c.b) == (1, 7)
        assert log == [
            (P.PRE_INIT, None),
            (P.ASSIGNING, "a"),
            (P.ASSIGNING, "b"),
            (P.POST_INIT, None),
        ]
        assert attr.get_init_phase(c) is P.READY
        assert attr.get_init_field(c) is None

    @pytest.mark.parametrize("slots", [True, False])
    def test_validating_phase(self, slots):
        seen = {}

        def v(self, a, value):
            seen["phase"] = attr.get_init_phase(self)
            seen["field"] = attr.get_init_field(self)

        class C:
            x = attrib(validator=v)
            y = attrib(validator=v)

        C = decorate(C, slots=slots)
        C(1, 2)

        assert seen == {"phase": P.VALIDATING, "field": None}

    @pytest.mark.parametrize("slots", [True, False])
    def test_field_order(self, slots):
        order = []

        def conv(v, self):
            order.append(attr.get_init_field(self))
            return v

        class C:
            first = attrib(converter=Converter(conv, takes_self=True))
            second = attrib(converter=Converter(conv, takes_self=True))
            third = attrib(default=0)

        C = decorate(C, slots=slots)
        C(1, 2)

        assert order == ["first", "second"]

    @pytest.mark.parametrize(
        "hook",
        [
            "converter",
            "factory",
            "validator",
            "pre_init",
            "post_init",
        ],
    )
    @pytest.mark.parametrize("slots", [True, False])
    def test_aborted_on_hook_error(self, slots, hook):
        class Boom(Exception):
            pass

        def raise_hook(self_or_v, *a):
            raise Boom

        class C:
            x = attrib(default=1)

        if hook == "converter":
            C.x = attrib(
                default=1,
                converter=Converter(raise_hook, takes_self=True),
            )
        elif hook == "factory":
            C.x = attrib(default=Factory(raise_hook, takes_self=True))
        elif hook == "validator":
            C.x = attrib(default=1, validator=raise_hook)

        if hook == "pre_init":

            def __attrs_pre_init__(self):
                raise Boom

            C.__attrs_pre_init__ = __attrs_pre_init__

        if hook == "post_init":

            def __attrs_post_init__(self):
                raise Boom

            C.__attrs_post_init__ = __attrs_post_init__

        C = decorate(C, slots=slots)
        captured = []

        if hook in ("converter", "factory", "validator"):
            orig = C.__init__

            def __init__(self, *a, **kw):
                captured.append(self)
                orig(self, *a, **kw)

            C.__init__ = __init__
        # Wrap the hooks so the half-built instance can be queried.
        elif hook == "pre_init":
            orig = C.__attrs_pre_init__

            def wrapped_pre(self):
                captured.append(self)
                orig(self)

            C.__attrs_pre_init__ = wrapped_pre
        else:
            orig = C.__attrs_post_init__

            def wrapped_post(self):
                try:
                    orig(self)
                finally:
                    captured.append(self)

            C.__attrs_post_init__ = wrapped_post

        with pytest.raises(Boom):
            C()

        inst = captured[0]
        assert attr.get_init_phase(inst) is P.ABORTED
        assert attr.get_init_phase(inst) is not P.READY

    def test_aborted_is_per_instance(self):
        class C:
            x = attrib(
                converter=Converter(
                    lambda v, self: 1 / 0 if v == "boom" else v,
                    takes_self=True,
                )
            )

        C = attr.attrs(C)
        ok = C(1)
        assert attr.get_init_phase(ok) is P.READY
        with pytest.raises(ZeroDivisionError):
            C("boom")
        # A later construction is unaffected.
        assert attr.get_init_phase(C(2)) is P.READY

    def test_user_init_calling_attrs_init(self):
        seen = []

        @attr.attrs(init=False)
        class C:
            x = attrib(
                converter=Converter(
                    lambda v, self: (
                        seen.append(
                            (
                                attr.get_init_phase(self),
                                attr.get_init_field(self),
                            )
                        )
                        or v
                    ),
                    takes_self=True,
                )
            )

            def __init__(self, val):
                # Before the generated flight starts, there is no flight.
                assert attr.get_init_phase(self) is P.READY
                self.__attrs_init__(val)

        c = C(5)
        assert c.x == 5
        assert seen == [(P.ASSIGNING, "x")]
        assert attr.get_init_phase(c) is P.READY

    def test_attrs_init_seals_too(self):
        @attr.attrs(init=False)
        class C:
            a = attrib()
            b = attrib(
                converter=Converter(
                    lambda v, self: object.__setattr__(self, "a", 999) or v,
                    takes_self=True,
                )
            )

            def __init__(self, a, b):
                self.__attrs_init__(a, b)

        with pytest.raises(SealedAttributeError):
            C(1, 2)


# ---------------------------------------------------------------------------
# (2) Sealing of already assigned fields
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("slots", [True, False])
@pytest.mark.parametrize("frozen", [True, False])
class TestSealing:
    def test_converter_rewrites_sealed_object_setattr(self, slots, frozen):
        class C:
            early = attrib()
            later = attrib(
                converter=Converter(
                    lambda v, self: (
                        object.__setattr__(self, "early", 999) or v
                    ),
                    takes_self=True,
                )
            )

        C = decorate(C, slots=slots, frozen=frozen)
        with pytest.raises(SealedAttributeError) as ei:
            C(1, 2)
        assert ei.value.name == "early"
        assert ei.value.phase == "assigning"

    def test_factory_init_false_rewrites_sealed(self, slots, frozen):
        class C:
            early = attrib()
            extra = attrib(
                init=False,
                default=Factory(
                    lambda self: object.__setattr__(self, "early", 42) or "x",
                    takes_self=True,
                ),
            )

        C = decorate(C, slots=slots, frozen=frozen)
        with pytest.raises(SealedAttributeError):
            C(1)

    def test_validator_rewrites_sealed(self, slots, frozen):
        class C:
            x = attrib(
                validator=lambda s, a, v: object.__setattr__(s, "x", v * 10)
            )

        C = decorate(C, slots=slots, frozen=frozen)
        with pytest.raises(SealedAttributeError) as ei:
            C(5)
        assert ei.value.phase == "validating"

    def test_reading_sealed_is_allowed(self, slots, frozen):
        seen = []

        class C:
            early = attrib()
            later = attrib(
                converter=Converter(
                    lambda v, self: seen.append(self.early) or v,
                    takes_self=True,
                )
            )

        C = decorate(C, slots=slots, frozen=frozen)
        c = C(11, 22)
        assert seen == [11]
        assert (c.early, c.later) == (11, 22)

    def test_happy_path_final_values(self, slots, frozen):
        class C:
            a = attrib(converter=lambda v: v * 2)
            b = attrib(
                converter=Converter(
                    lambda v, self: v + self.a, takes_self=True
                )
            )
            c = attrib(default=Factory(lambda: 3))

        C = decorate(C, slots=slots, frozen=frozen)
        instance = C(1, 0)
        assert (instance.a, instance.b, instance.c) == (2, 2, 3)
        assert attr.get_init_phase(instance) is P.READY

    def test_non_attribute_names_unaffected(self, slots, frozen):
        if slots:
            pytest.skip("pure slotted instances have no __dict__")

        class C:
            x = attrib(
                converter=Converter(
                    lambda v, self: (
                        object.__setattr__(self, "other", 123) or v
                    ),
                    takes_self=True,
                )
            )

        C = decorate(C, slots=slots, frozen=frozen)
        assert C(1).other == 123

    def test_in_place_mutation_of_sealed_value_allowed(self, slots, frozen):
        class C:
            items = attrib(
                converter=Converter(
                    lambda v, self: (v.append(2), v)[1], takes_self=True
                )
            )

        C = decorate(C, slots=slots, frozen=frozen)
        c = C([1])
        assert c.items == [1, 2]

    def test_deleting_sealed_field_fails(self, slots, frozen):
        class C:
            a = attrib()
            b = attrib(
                converter=Converter(
                    lambda v, self: object.__delattr__(self, "a") or v,
                    takes_self=True,
                )
            )

        C = decorate(C, slots=slots, frozen=frozen)
        with pytest.raises(SealedAttributeError):
            C(1, 2)


class TestSealingGeneratedSetattr:
    def test_hook_uses_generated_setattr_sealed(self):
        class C:
            early = attrib()
            later = attrib(
                converter=Converter(
                    lambda v, self: setattr(self, "early", 999) or v,
                    takes_self=True,
                )
            )

        # on_setattr produces a generated __setattr__ on a non-frozen class.
        C = attr.attrs(C, on_setattr=attr.setters.NO_OP)
        with pytest.raises(SealedAttributeError):
            C(1, 2)

    def test_hook_uses_generated_setattr_unassigned(self):
        class C:
            early = attrib(
                converter=Converter(
                    lambda v, self: setattr(self, "later", 777) or v,
                    takes_self=True,
                )
            )
            later = attrib(default=0)

        C = attr.attrs(C, on_setattr=attr.setters.NO_OP)
        with pytest.raises(UnassignedAttributeError):
            C(1)

    @staticmethod
    def _identity_setattr(self_, a, v):
        return v

    def test_real_generated_setattr_sealed(self):
        class C:
            early = attrib(on_setattr=self._identity_setattr)
            later = attrib(
                converter=Converter(
                    lambda v, self: setattr(self, "early", 999) or v,
                    takes_self=True,
                )
            )

        C = attr.attrs(C)
        with pytest.raises(SealedAttributeError) as ei:
            C(1, 2)
        assert ei.value.name == "early"

    def test_real_generated_setattr_unassigned(self):
        class C:
            early = attrib(
                on_setattr=self._identity_setattr,
                converter=Converter(
                    lambda v, self: setattr(self, "later", 777) or v,
                    takes_self=True,
                ),
            )
            later = attrib(on_setattr=self._identity_setattr, default=0)

        C = attr.attrs(C)
        with pytest.raises(UnassignedAttributeError) as ei:
            C(1)
        assert ei.value.name == "later"

    def test_real_generated_setattr_pre_init(self):
        class C:
            x = attrib(
                on_setattr=self._identity_setattr,
                converter=Converter(lambda v: v, takes_self=False),
            )

            def __attrs_pre_init__(self):
                self.x = 999

        C = attr.attrs(C)
        with pytest.raises(UnassignedAttributeError) as ei:
            C(1)
        assert ei.value.phase == "pre_init"

    def test_real_generated_setattr_current_field_allowed(self):
        # A converter may assign its OWN field through the generated
        # __setattr__; the planned write seals the converter's return value.
        class C:
            x = attrib(
                on_setattr=self._identity_setattr,
                converter=Converter(
                    lambda v, self: setattr(self, "x", 123) or v,
                    takes_self=True,
                ),
            )
            later = attrib(
                converter=Converter(
                    lambda v, self: setattr(self, "x", 999) or v,
                    takes_self=True,
                )
            )

        C = attr.attrs(C)
        with pytest.raises(SealedAttributeError):
            C(1, 2)

        class D:
            x = attrib(
                on_setattr=self._identity_setattr,
                converter=Converter(
                    lambda v, self: setattr(self, "x", 123) or v,
                    takes_self=True,
                ),
            )

        D = attr.attrs(D)
        assert D(5).x == 5


# ---------------------------------------------------------------------------
# (3) Early writes to not-yet-assigned fields
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("slots", [True, False])
@pytest.mark.parametrize("frozen", [True, False])
class TestNoEarlyWrite:
    def test_converter_pre_writes_later(self, slots, frozen):
        class C:
            early = attrib(
                converter=Converter(
                    lambda v, self: (
                        object.__setattr__(self, "later", 777) or v
                    ),
                    takes_self=True,
                )
            )
            later = attrib(default=0)

        C = decorate(C, slots=slots, frozen=frozen)
        with pytest.raises(UnassignedAttributeError) as ei:
            C(1)
        assert ei.value.name == "later"

    def test_factory_branch_pre_writes_later(self, slots, frozen):
        class C:
            early = attrib(
                default=Factory(
                    lambda self: object.__setattr__(self, "later", 777) or 1,
                    takes_self=True,
                )
            )
            later = attrib(default=0)

        C = decorate(C, slots=slots, frozen=frozen)
        with pytest.raises(UnassignedAttributeError):
            C()

    def test_pre_writes_init_false_takes_self_factory(self, slots, frozen):
        class C:
            early = attrib(
                converter=Converter(
                    lambda v, self: (
                        object.__setattr__(self, "extra", "e") or v
                    ),
                    takes_self=True,
                )
            )
            extra = attrib(
                init=False,
                default=Factory(lambda self: "f", takes_self=True),
            )

        C = decorate(C, slots=slots, frozen=frozen)
        with pytest.raises(UnassignedAttributeError):
            C(1)

    def test_init_false_takes_self_factory_sealed_afterwards(
        self, slots, frozen
    ):
        class C:
            early = attrib()
            extra = attrib(
                init=False,
                default=Factory(lambda self: "f", takes_self=True),
            )
            later = attrib(
                converter=Converter(
                    lambda v, self: (
                        object.__setattr__(self, "extra", "x") or v
                    ),
                    takes_self=True,
                )
            )

        C = decorate(C, slots=slots, frozen=frozen)
        with pytest.raises(SealedAttributeError):
            C(1, 2)

    def test_pre_writes_init_false_field(self, slots, frozen):
        class C:
            early = attrib(
                converter=Converter(
                    lambda v, self: (
                        object.__setattr__(self, "extra", "e") or v
                    ),
                    takes_self=True,
                )
            )
            extra = attrib(init=False, default="d")

        C = decorate(C, slots=slots, frozen=frozen)
        with pytest.raises(UnassignedAttributeError):
            C(1)

    def test_subclass_converter_attacks_base_and_own_fields(
        self, slots, frozen
    ):
        class Base:
            base_a = attrib()
            base_b = attrib()

        Base = attr.attrs(Base)

        class Sub(Base):
            sub_c = attrib(
                converter=Converter(
                    lambda v, self: (
                        object.__setattr__(self, "base_a", 999) or v
                    ),
                    takes_self=True,
                )
            )
            sub_d = attrib(default=0)

        Sub = attr.attrs(Sub, slots=slots, frozen=frozen)
        with pytest.raises(SealedAttributeError):
            Sub(1, 2, 3)

        class Sub2(Base):
            sub_c = attrib()
            sub_d = attrib(
                converter=Converter(
                    lambda v, self: object.__setattr__(self, "sub_c", 9) or v,
                    takes_self=True,
                )
            )

        Sub2 = attr.attrs(Sub2, slots=slots, frozen=frozen)
        with pytest.raises(SealedAttributeError):
            Sub2(1, 2, 3, 4)


# ---------------------------------------------------------------------------
# (4) pre_init
# ---------------------------------------------------------------------------


class TestPreInit:
    @pytest.mark.parametrize("slots", [True, False])
    @pytest.mark.parametrize("frozen", [True, False])
    def test_pre_init_cannot_write(self, slots, frozen):
        class C:
            x = attrib(converter=Converter(lambda v, self: v, takes_self=True))

            def __attrs_pre_init__(self):
                object.__setattr__(self, "x", 999)

        C = decorate(C, slots=slots, frozen=frozen)
        with pytest.raises(UnassignedAttributeError) as ei:
            C(1)
        assert ei.value.phase == "pre_init"

    @pytest.mark.parametrize("slots", [True, False])
    def test_pre_init_receives_raw_arguments(self, slots):
        seen = {}

        class C:
            x = attrib(converter=lambda v: v * 10)

            def __attrs_pre_init__(self, x):
                seen["x"] = x

        C = decorate(C, slots=slots)
        c = C(3)
        assert seen == {"x": 3}
        assert c.x == 30

    def test_pre_init_phase_query(self):
        class C:
            x = attrib()

            def __attrs_pre_init__(self):
                assert attr.get_init_phase(self) is P.PRE_INIT
                assert attr.get_init_field(self) is None

        attr.attrs(C)(1)


# ---------------------------------------------------------------------------
# (5) post_init lifting and ready semantics
# ---------------------------------------------------------------------------


class TestPostInit:
    @pytest.mark.parametrize("slots", [True, False])
    def test_frozen_post_init_object_setattr_pin(self, slots):
        class C:
            x = attrib()

            def __attrs_post_init__(self):
                assert attr.get_init_phase(self) is P.POST_INIT
                object.__setattr__(self, "x", self.x + 1)

        C = attr.attrs(C, slots=slots, frozen=True)
        c = C(1)
        assert c.x == 2
        assert attr.get_init_phase(c) is P.READY

    @pytest.mark.parametrize("slots", [True, False])
    def test_nonfrozen_post_init_through_on_setattr(self, slots):
        calls = []

        def hook(self, a, v):
            calls.append((a.name, v))
            return v

        class C:
            x = attrib(on_setattr=hook)

            def __attrs_post_init__(self):
                self.x = self.x + 1

        C = attr.attrs(C, slots=slots)
        c = C(1)
        assert c.x == 2
        # One call from post-init through the generated __setattr__.
        assert ("x", 2) in calls

    def test_ready_frozen_raises_frozen_error_not_sealing(self):
        @attr.s(frozen=True, slots=True)
        class C:
            x = attrib()

        c = C(1)
        with pytest.raises(FrozenInstanceError):
            c.x = 2
        # The pin: frozen is enforced through the generated __setattr__;
        # object.__setattr__ bypass is unchanged (and must not be confused
        # with a construction sealing error either way).
        object.__setattr__(c, "x", 3)
        assert c.x == 3

    def test_ready_on_setattr_semantics(self):
        calls = []

        @attr.s(on_setattr=lambda self, a, v: calls.append(v) or v)
        class C:
            x = attrib()

        c = C(1)
        c.x = 5
        assert c.x == 5
        assert calls == [5]

    def test_post_init_failure_aborts(self):
        box = {}

        @attr.s
        class C:
            x = attrib()

            def __attrs_post_init__(self):
                box["inst"] = self
                raise RuntimeError

        with pytest.raises(RuntimeError):
            C(1)
        assert attr.get_init_phase(box["inst"]) is P.ABORTED


# ---------------------------------------------------------------------------
# (6) The three forks: evolve / assoc / setstate
# ---------------------------------------------------------------------------


class TestForks:
    def test_evolve_reflies_converters_run(self):
        counter = []

        @attr.s
        class C:
            x = attrib(converter=lambda v: counter.append(v) or v + 1)
            y = attrib(converter=lambda v: counter.append(v) or v - 1)

        c1 = C(1, 10)
        c2 = attr.evolve(c1, x=100)
        # Both fields are converted again on the new instance, including
        # unchanged y.
        assert counter == [1, 10, 100, 9]
        assert (c1.x, c1.y) == (2, 9)
        assert (c2.x, c2.y) == (101, 8)
        assert attr.get_init_phase(c2) is P.READY

    def test_evolve_phase_is_observable(self):
        seen = []

        def conv(v, self):
            seen.append((attr.get_init_phase(self), attr.get_init_field(self)))
            return v

        @attr.s
        class C:
            x = attrib(converter=Converter(conv, takes_self=True))

        attr.evolve(C(1), x=2)
        assert seen[-1] == (P.ASSIGNING, "x")

    @pytest.mark.skipif(not PY_3_13_PLUS, reason="3.13+ __replace__")
    def test_replace_reflies(self):
        counter = []

        @attr.s
        class C:
            x = attrib(converter=lambda v: counter.append(v) or v)

        c1 = C(1)
        c2 = copy.replace(c1, x=2)
        assert counter == [1, 2]
        assert attr.get_init_phase(c2) is P.READY

    def test_assoc_no_flight_ready_storage_names(self):
        counter = []

        @attr.s
        class C:
            x = attrib(converter=lambda v: counter.append(v) or v)

        c1 = C(1)
        c2 = attr.assoc(c1, x=9)
        assert c2.x == 9
        # Converters do NOT run again.
        assert counter == [1]
        assert attr.get_init_phase(c2) is P.READY

    @pytest.mark.parametrize("cls", [_PickleSlotted, _PickleDict])
    def test_pickle_ready_no_flight(self, cls):
        _pickle_convert_calls.clear()
        _pickle_self_phases.clear()

        c1 = cls(1, 2)
        assert _pickle_self_phases == [P.ASSIGNING]
        assert len(_pickle_convert_calls) == 2
        _pickle_convert_calls.clear()

        c2 = pickle.loads(pickle.dumps(c1))
        assert (c2.x, c2.y) == (1, 2)
        # __setstate__ restores storage directly: no converter runs again
        # and the restored instance never enters a construction flight.
        assert _pickle_convert_calls == []
        assert _pickle_self_phases == [P.ASSIGNING]
        assert attr.get_init_phase(c2) is P.READY
        assert attr.get_init_field(c2) is None

    def test_slotted_copy_ready(self):
        c = pickle.loads(pickle.dumps(_PickleSlotted(4, 5)))
        assert (c.x, c.y) == (4, 5)
        assert attr.get_init_phase(c) is P.READY

    def test_forks_are_distinct_mechanisms(self):
        # evolve must still be cls(**changes), assoc copy+setattr: the APIs
        # exist independently and behave differently w.r.t. converters.
        @attr.s
        class C:
            x = attrib(converter=lambda v: v + 1)

        c = C(1)
        assert attr.evolve(c).x == 3  # converted again: 2 -> 3
        assert attr.assoc(c, x=42).x == 42  # raw storage write


# ---------------------------------------------------------------------------
# (7) Split key spaces: storage name vs alias
# ---------------------------------------------------------------------------


class TestKeySpaces:
    def test_seal_uses_storage_name_evolve_uses_alias(self):
        C = attr.make_class(
            "C",
            {
                "priv": attrib(alias="pub"),
                "other": attrib(
                    alias="o",
                    converter=Converter(
                        lambda v, self: (
                            object.__setattr__(self, "priv", 999) or v
                        ),
                        takes_self=True,
                    ),
                ),
            },
        )
        with pytest.raises(SealedAttributeError) as ei:
            C(pub=1, o=2)
        assert ei.value.name == "priv"

        class D:
            priv = attrib(alias="pub")

        D = attr.attrs(D)
        d = D(pub=1)
        assert attr.fields(D)[0].name == "priv"
        assert attr.fields(D)[0].alias == "pub"
        assert attr.evolve(d, pub=7).priv == 7
        assert attr.assoc(d, priv=8).priv == 8
        assert attr.get_init_field(D(pub=1)) is None

    def test_private_attribute_double_underscore(self):
        @attr.s
        class C:
            __v = attrib()

        fld = attr.fields(C)[0]
        assert fld.name != fld.alias

        c = C(**{fld.alias: 1})
        assert getattr(c, fld.name) == 1
        assert attr.evolve(c, **{fld.alias: 2}) != c
        assert getattr(attr.evolve(c, **{fld.alias: 2}), fld.name) == 2
        assert getattr(attr.assoc(c, **{fld.name: 3}), fld.name) == 3


# ---------------------------------------------------------------------------
# (8) Coverage: generators, inheritance storage mix, threads
# ---------------------------------------------------------------------------


class TestCoverage:
    def test_autoattribs_and_make_class(self):
        @attr.s(auto_attribs=True)
        class Auto:
            early: int
            later: int = attrib(
                default=0,
                converter=Converter(
                    lambda v, self: (
                        object.__setattr__(self, "early", 999) or v
                    ),
                    takes_self=True,
                ),
            )

        with pytest.raises(SealedAttributeError):
            Auto(1)

        MC = attr.make_class(
            "MC",
            {
                "early": attrib(),
                "later": attrib(
                    converter=Converter(
                        lambda v, self: (
                            object.__setattr__(self, "early", 999) or v
                        ),
                        takes_self=True,
                    )
                ),
            },
        )
        with pytest.raises(SealedAttributeError):
            MC(1, 2)

    def test_inheritance_order_base_first(self):
        order = []

        def conv(v, self):
            order.append(attr.get_init_field(self))
            return v

        class Base:
            b1 = attrib(converter=Converter(conv, takes_self=True))
            b2 = attrib(converter=Converter(conv, takes_self=True))

        Base = attr.attrs(Base)

        class Sub(Base):
            s1 = attrib(converter=Converter(conv, takes_self=True))

        Sub = attr.attrs(Sub)
        Sub(1, 2, 3)
        assert order == ["b1", "b2", "s1"]

    def test_mixed_storage_dict_base_slots_sub(self):
        @attr.s
        class Base:
            a = attrib()

        @attr.s(slots=True)
        class Sub(Base):
            b = attrib(
                converter=Converter(
                    lambda v, self: object.__setattr__(self, "a", -1) or v,
                    takes_self=True,
                )
            )
            c = attrib()

        # Base field lives in __dict__, sub fields in slots: sealing must
        # cover both storage kinds.
        with pytest.raises(SealedAttributeError):
            Sub(1, 2, 3)

    def test_mixed_storage_slots_base_dict_sub(self):
        @attr.s(slots=True)
        class Base:
            a = attrib()

        @attr.s
        class Sub(Base):
            b = attrib(
                converter=Converter(
                    lambda v, self: object.__setattr__(self, "a", -1) or v,
                    takes_self=True,
                )
            )

        with pytest.raises(SealedAttributeError):
            Sub(1, 2)

    def test_takes_field_converter_under_flight(self):
        def conv(v, field):
            return v + 1

        class C:
            x = attrib(
                validator=lambda *a: None,  # makes the flight active
                converter=Converter(conv, takes_field=True),
            )

        C = attr.attrs(C)
        assert C(1).x == 2

    def test_slots_without_weakref_slot_abort_then_rebuild(self):
        class C:
            x = attrib(
                converter=Converter(
                    lambda v, self: 1 / 0 if v == "x" else v,
                    takes_self=True,
                )
            )

        C = attr.attrs(C, slots=True, weakref_slot=False)
        with pytest.raises(ZeroDivisionError):
            C("x")
        # Another instance builds cleanly and observes its own flight.
        seen = []

        class D:
            y = attrib(
                converter=Converter(
                    lambda v, self: (
                        seen.append(attr.get_init_field(self)) or v
                    ),
                    takes_self=True,
                )
            )

        D = attr.attrs(D, slots=True, weakref_slot=False)
        d = D(1)
        assert d.y == 1
        assert seen == ["y"]
        assert attr.get_init_phase(d) is P.READY

    def test_two_threads_independent_flights(self):
        barrier = threading.Barrier(2)
        errors = []

        def conv(v, self):
            # Both threads park while each instance assigns its own field.
            barrier.wait(timeout=5)
            return attr.get_init_field(self), v

        @attr.s
        class C:
            x = attrib(converter=Converter(conv, takes_self=True))

        results = {}

        def worker(key):
            try:
                results[key] = C(key)
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [
            threading.Thread(target=worker, args=(key,)) for key in ("a", "b")
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert results["a"].x == ("x", "a")
        assert results["b"].x == ("x", "b")
        assert attr.get_init_phase(results["a"]) is P.READY
        assert attr.get_init_phase(results["b"]) is P.READY

    def test_cross_thread_cannot_see_current_field(self):
        ready = threading.Event()
        release = threading.Event()
        box = {}

        def conv(v, self):
            box["inst"] = self
            ready.set()
            assert release.wait(5)
            return v

        @attr.s
        class C:
            x = attrib(converter=Converter(conv, takes_self=True))

        t = threading.Thread(target=lambda: box.update(result=C(1)))
        t.start()
        assert ready.wait(5)

        building = box["inst"]
        # The phase itself is observable, but the assigning storage name is
        # never reported to a different thread.
        assert attr.get_init_phase(building) is P.ASSIGNING
        assert attr.get_init_field(building) is None

        release.set()
        t.join()
        assert box["result"].x == 1
        assert attr.get_init_phase(box["result"]) is P.READY

    def test_user_getattr_cannot_spoof_unassigned_slots(self):
        # A slotted class whose __getattr__ fabricates values for unset
        # slots must not fool the unassigned-field check.
        @attr.s(slots=True)
        class C:
            a = attrib()
            b = attrib(converter=Converter(lambda v, self: v, takes_self=True))
            c = attrib(default=3)
            d = attrib(default=4)

            def __getattr__(self, name):
                return 123456

        c = C(1, 2)
        assert (c.a, c.b, c.c, c.d) == (1, 2, 3, 4)
        assert attr.get_init_phase(c) is P.READY

    def test_aborted_weakrefable_entry_cleaned_on_gc(self):
        from attr import _flight

        @attr.s
        class C:
            x = attrib(
                converter=Converter(lambda v, self: 1 / 0, takes_self=True)
            )

        with pytest.raises(ZeroDivisionError):
            C(1)
        # The abandoned instance is still referenced by the traceback of
        # the caught error; drop everything and force collection.
        gc.collect()
        # Every lingering entry must still own a live instance: once the
        # instance dies, the weakref finalizer must have removed its entry
        # (otherwise an id reuse could report a bogus aborted/ready phase).
        for key, flight in list(_flight._flights.items()):
            if flight._ref is not None:
                assert flight._ref() is not None, key
            else:
                # Non-weakrefable slotted classes intentionally keep a
                # strong reference to the abandoned half-built instance.
                assert flight._strong is not None

    def test_aborted_instance_queries_aborted_until_collected(self):
        box = {}

        @attr.s
        class C:
            x = attrib(
                converter=Converter(
                    lambda v, self: box.__setitem__("inst", self) or 1 / 0,
                    takes_self=True,
                )
            )

        with pytest.raises(ZeroDivisionError):
            C(1)
        inst = box["inst"]
        assert attr.get_init_phase(inst) is P.ABORTED
        assert attr.get_init_phase(inst) is not P.READY
        assert attr.get_init_field(inst) is None
