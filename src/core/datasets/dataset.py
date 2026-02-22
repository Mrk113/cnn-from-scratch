"""Dataset base class.

This module defines the interface expected from dataset implementations. 
"""

from typing import Any, Callable, Optional


class DataSet:
    """Dataset base class.

    Subclasses should supply storage for data and implement length and item
    retrieval to integrate with training and evaluation loops.
    """

    def __init__(self,
                 *,
                 root: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 ) -> None: 
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        """Return the dataset size.

        Returns:
            int: Number of samples available.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError

    def __getitem__(self, index: int) -> Any:
        """Retrieve the item at the given index.

        Args:
            index: Zero-based index of the desired sample.

        Returns:
            Any: Sample and target pair or sample object, as defined by subclass.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError