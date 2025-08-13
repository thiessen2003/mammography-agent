from typing import TypedDict, List, Dict, Any, Optional


class UserInput(TypedDict):
    """Type definition for UserInput structure."""
    username: str
    image: str


class UserInputDTO:
    """Data Transfer Object for mapping user input from JSON."""
    
    def __init__(self, username: str, image: str, **kwargs):
        self.username = username
        self.image = image
        self._metadata = kwargs
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value by key, similar to dict.get()."""
        if hasattr(self, key):
            return getattr(self, key)
        return self._metadata.get(key, default)
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access."""
        if hasattr(self, key):
            return getattr(self, key)
        return self._metadata[key]
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        return hasattr(self, key) or key in self._metadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = {
            'username': self.username,
            'image': self.image
        }
        result.update(self._metadata)
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserInputDTO':
        """Create UserInputDTO instance from dictionary/JSON data."""
        username = data.get('username', '')
        image = data.get('image', '')
        return cls(username=username, image=image, **data)
    
    def add_metadata(self, key: str, value: Any) -> None:
        """Add custom metadata field."""
        self._metadata[key] = value
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get custom metadata field value."""
        return self._metadata.get(key, default)
    
    def __repr__(self) -> str:
        """String representation of the object."""
        return f"UserInputDTO(username='{self.username}', image='{self.image}', metadata={self._metadata})"

