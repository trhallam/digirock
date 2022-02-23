import pytest

from digirock._exceptions import WorkflowError, PrototypeError


def test_WorkflowError():
    with pytest.raises(WorkflowError):
        raise WorkflowError("workflow", "a msg")


def test_WorkflowError_print():
    try:
        raise WorkflowError(("a", "b"), "a msg")
    except WorkflowError as err:
        print(err)

    try:
        raise WorkflowError("a", "a msg")
    except WorkflowError as err:
        print(err)


def test_PrototypeError():
    with pytest.raises(PrototypeError):
        raise PrototypeError("ptye", "a msg")


def test_PrototypeError_print():
    try:
        raise PrototypeError("a", "a msg")
    except PrototypeError as err:
        print(err)

    try:
        raise PrototypeError(("a", "b"), "a msg")
    except PrototypeError as err:
        print(err)
