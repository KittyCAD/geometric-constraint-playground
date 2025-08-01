class ConflictError(ValueError):
    pass


class UnsupportedPrimitiveError(NotImplementedError):
    def __init__(self, primitive_type: str):
        super().__init__(
            f"Primitive type {primitive_type} is not supported in this context."
        )
        self.primitive_type = primitive_type
