import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import GameState, ActionResponse, ModelInfo
from agent.model import DQN

# --- Config ---
MODEL_PATH  = "models/best_model.pth"
INPUT_SIZE  = 11
OUTPUT_SIZE = 3
ACTION_NAMES = ["straight", "right", "left"]

# --- Load model once at startup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = DQN(input_size=INPUT_SIZE, output_size=OUTPUT_SIZE).to(device)

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"Model loaded from {MODEL_PATH}")
except FileNotFoundError:
    raise RuntimeError(f"Model not found at {MODEL_PATH}. Train first.")

# --- FastAPI app ---
app = FastAPI(
    title="RL Snake Bot API",
    description="Serves a trained DQN agent that plays Snake.",
    version="1.0.0",
)

# Allow React frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "message": "RL Snake Bot API is running"
    }


@app.get("/info", response_model=ModelInfo)
def model_info():
    return ModelInfo(
        model="Deep Q-Network (DQN)",
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
        actions=ACTION_NAMES,
        model_path=MODEL_PATH,
    )


@app.post("/predict", response_model=ActionResponse)
def predict(game_state: GameState):
    state = game_state.state

    # Validate input
    if len(state) != INPUT_SIZE:
        raise HTTPException(
            status_code=422,
            detail=f"State must have {INPUT_SIZE} values, got {len(state)}"
        )

    # Run inference
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = model(state_tensor)

    action     = q_values.argmax().item()
    q_list     = q_values.squeeze().tolist()

    return ActionResponse(
        action=action,
        action_name=ACTION_NAMES[action],
        q_values=q_list,
    )