import { useState, useRef, useCallback } from 'react'
import axios from 'axios'
import type { GameState, Action } from '../types'

const BLOCK  = 20
const WIDTH  = 480
const HEIGHT = 480
const API    = 'http://127.0.0.1:8000'

function initState(): GameState {
  return {
    snake: [
      { x: 240, y: 240 },
      { x: 220, y: 240 },
      { x: 200, y: 240 },
    ],
    food: randomFood([{ x: 240, y: 240 }]),
    direction: 'RIGHT',
    score: 0,
    gameOver: false,
  }
}

function randomFood(snake: { x: number; y: number }[]) {
  let food: { x: number; y: number }
  do {
    food = {
      x: Math.floor(Math.random() * (WIDTH / BLOCK)) * BLOCK,
      y: Math.floor(Math.random() * (HEIGHT / BLOCK)) * BLOCK,
    }
  } while (snake.some(s => s.x === food.x && s.y === food.y))
  return food
}

function getStateVector(game: GameState): number[] {
  const head = game.snake[0]
  const dir  = game.direction

  const pL = { x: head.x - BLOCK, y: head.y }
  const pR = { x: head.x + BLOCK, y: head.y }
  const pU = { x: head.x, y: head.y - BLOCK }
  const pD = { x: head.x, y: head.y + BLOCK }

  const isCollision = (p: { x: number; y: number }) =>
    p.x < 0 || p.x >= WIDTH || p.y < 0 || p.y >= HEIGHT ||
    game.snake.slice(1).some(s => s.x === p.x && s.y === p.y)

  const dR = dir === 'RIGHT', dL = dir === 'LEFT'
  const dU = dir === 'UP',    dD = dir === 'DOWN'

  return [
    // Danger straight
    Number((dR && isCollision(pR)) || (dL && isCollision(pL)) ||
           (dU && isCollision(pU)) || (dD && isCollision(pD))),
    // Danger right
    Number((dU && isCollision(pR)) || (dD && isCollision(pL)) ||
           (dL && isCollision(pU)) || (dR && isCollision(pD))),
    // Danger left
    Number((dD && isCollision(pR)) || (dU && isCollision(pL)) ||
           (dR && isCollision(pU)) || (dL && isCollision(pD))),
    // Direction
    Number(dL), Number(dR), Number(dU), Number(dD),
    // Food
    Number(game.food.x < head.x),
    Number(game.food.x > head.x),
    Number(game.food.y < head.y),
    Number(game.food.y > head.y),
  ]
}

function applyAction(game: GameState, action: Action): GameState {
  const clockWise: GameState['direction'][] = ['RIGHT', 'DOWN', 'LEFT', 'UP']
  const idx = clockWise.indexOf(game.direction)

  let newDir = game.direction
  if (action === 1) newDir = clockWise[(idx + 1) % 4]
  else if (action === 2) newDir = clockWise[(idx + 3) % 4]

  const head = game.snake[0]
  let newHead = { ...head }
  if (newDir === 'RIGHT') newHead.x += BLOCK
  else if (newDir === 'LEFT')  newHead.x -= BLOCK
  else if (newDir === 'UP')    newHead.y -= BLOCK
  else if (newDir === 'DOWN')  newHead.y += BLOCK

  // Collision check
  const hitWall = newHead.x < 0 || newHead.x >= WIDTH ||
                  newHead.y < 0 || newHead.y >= HEIGHT
  const hitSelf = game.snake.some(s => s.x === newHead.x && s.y === newHead.y)

  if (hitWall || hitSelf) {
    return { ...game, gameOver: true }
  }

  const newSnake = [newHead, ...game.snake]
  let newScore   = game.score
  let newFood    = game.food

  if (newHead.x === game.food.x && newHead.y === game.food.y) {
    newScore += 1
    newFood   = randomFood(newSnake)
  } else {
    newSnake.pop()
  }

  return {
    snake: newSnake,
    food: newFood,
    direction: newDir,
    score: newScore,
    gameOver: false,
  }
}

export function useSnakeGame() {
  const [game, setGame]       = useState<GameState>(initState())
  const [running, setRunning] = useState(false)
  const [episode, setEpisode] = useState(1)
  const [bestScore, setBestScore] = useState(0)
  const [qValues, setQValues] = useState<number[]>([0, 0, 0])
  const loopRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const gameRef = useRef<GameState>(game)
  gameRef.current = game

  const step = useCallback(async () => {
    if (gameRef.current.gameOver) {
      // Start new episode
      setBestScore(prev => Math.max(prev, gameRef.current.score))
      setEpisode(prev => prev + 1)
      setGame(initState())
      return
    }

    try {
      const stateVec = getStateVector(gameRef.current)
      const res = await axios.post(`${API}/predict`, { state: stateVec })
      const action: Action = res.data.action
      setQValues(res.data.q_values)
      setGame(prev => applyAction(prev, action))
    } catch {
      setRunning(false)
    }
  }, [])

  const start = useCallback(() => {
    setRunning(true)
    const loop = () => {
      step()
      loopRef.current = setTimeout(loop, 100) // 10 FPS
    }
    loopRef.current = setTimeout(loop, 100)
  }, [step])

  const stop = useCallback(() => {
    setRunning(false)
    if (loopRef.current) clearTimeout(loopRef.current)
  }, [])

  const reset = useCallback(() => {
    stop()
    setGame(initState())
    setEpisode(1)
    setBestScore(0)
    setQValues([0, 0, 0])
  }, [stop])

  return { game, running, episode, bestScore, qValues, start, stop, reset }
}