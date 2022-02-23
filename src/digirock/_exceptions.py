"""Common Exceptions for digirock functions and classes.
"""


class Error(Exception):
    """Base class for excceptions in this module."""

    pass


class WorkflowError(Error):
    """Exception raised for errors where the defined workflow has not been followed
    correctly.

    Attributes:
        expression  -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, name, message):
        """Constructor

        Args:
            name (object or tuple): Where this error was raised and parents names
            message (string): Which part of the workflow wasn't followed and
            what to do.
        """
        self.name = name
        self.message = message

    def __str__(self):

        if isinstance(self.name, tuple):
            insert = " in ".join(self.name)
        else:
            insert = self.name
        return ("Workflow for {!r} has not been followed, {!r}").format(
            insert, self.message
        )


class PrototypeError(Error):
    """Exception raised when trying to call a method proto-type in a class that doesn't actually have
    a call, the class needs to be sub-classed and the method overwritten.
    """

    def __init__(self, name, message):
        """Constructor

        Args:
            name (object or tuple): Where this error was raised and parents names
            message (string): Which part of the workflow wasn't followed and
            what to do.
        """
        self.name = name
        self.message = message

    def __str__(self):

        if isinstance(self.name, tuple):
            insert = " in ".join(self.name)
        else:
            insert = self.name
        return (
            "This is a prototype method for {!r}, use a child class instead of the baseclass {!r}."
        ).format(self.message, insert)
