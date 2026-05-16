export interface GameState {
  snake: { x: number; y: number }[]
  food: { x: number; y: number }
  direction: 'RIGHT' | 'LEFT' | 'UP' | 'DOWN'
  score: number
  gameOver: boolean
}

export interface PredictResponse {
  action: number
  action_name: string
  q_values: number[]
}

export type Action = 0 | 1 | 2