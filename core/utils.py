from dataclasses import dataclass


@dataclass(slots=True)
class Media:
    title: str
    hashed: str
