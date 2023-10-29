class NetworkOutput:
    def __init__(self, res: float = 0) -> None:
        self.id = 'O/?/?'
        self.previous_outputs: 'list[float]' = []
        self.output: float = res

    def set_id(self, layer: int, position: int) -> None:
        self.id: str = f'O/{layer}/{position}'

    @property
    def output(self) -> float:
        return self.__output

    @output.setter
    def output(self, value: float) -> None:
        if not isinstance(value, (int, float)):
            raise ValueError(f"Incorrect data detected {self.id} | {value}")
        self.__output = value
        self.previous_outputs.append(self.__output)
        if len(self.previous_outputs) > 50:
            self.previous_outputs.pop(0)

    def __repr__(self) -> str:
        return self.id
