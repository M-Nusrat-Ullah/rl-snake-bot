from pydantic import BaseModel
from typing import List


class GameState(BaseModel):
    """
    The 11-value state vector from get_state().
    Sent by the client to get the next action.
    """
    state: List[int]

    class Config:
        json_schema_extra = {
            "example": {
                "state": [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0]
            }
        }


class ActionResponse(BaseModel):
    """Response from /predict endpoint."""
    action: int          # 0=straight, 1=right, 2=left
    action_name: str     # human readable
    q_values: List[float]


class ModelInfo(BaseModel):
    """Response from /info endpoint."""
    model: str
    input_size: int
    output_size: int
    actions: List[str]
    model_path: str