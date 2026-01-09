from typing import Any, Callable, Optional

class DataSet:
    """Base class for datasets."""
    def __init__(self,
                 *,
                 root: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 ) -> None: 
        # Provided by subclasses
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index: int) -> Any:
        # Returns the item at the specified index
        raise NotImplementedError