from dataclasses import dataclass


@dataclass()
class EpochResult:
    train: None
    val: None
    test: None

    def __repr__(self):
        return ""



if __name__ == '__main__':
    pass