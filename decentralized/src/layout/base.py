from abc import ABC, abstractclassmethod


class BaseLayout(ABC):
    @property
    @abstractclassmethod
    def value():
        raise ValueError("value not implemented!")

    @property
    @abstractclassmethod
    def label():
        raise ValueError("label not implemented!")
