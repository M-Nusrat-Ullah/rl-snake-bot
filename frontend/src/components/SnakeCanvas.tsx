import { useEffect, useRef } from 'react'
import type { GameState } from '../types'

const BLOCK  = 20

interface Props {
  game: GameState
  width?: number
  height?: number
}

export default function SnakeCanvas({ game, width = 480, height = 480 }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Background
    ctx.fillStyle = '#111827'
    ctx.fillRect(0, 0, width, height)

    // Grid lines (subtle)
    ctx.strokeStyle = '#1f2937'
    ctx.lineWidth = 0.5
    for (let x = 0; x < width; x += BLOCK) {
      ctx.beginPath()
      ctx.moveTo(x, 0)
      ctx.lineTo(x, height)
      ctx.stroke()
    }
    for (let y = 0; y < height; y += BLOCK) {
      ctx.beginPath()
      ctx.moveTo(0, y)
      ctx.lineTo(width, y)
      ctx.stroke()
    }

    // Food
    ctx.fillStyle = '#ef4444'
    ctx.beginPath()
    ctx.arc(
      game.food.x + BLOCK / 2,
      game.food.y + BLOCK / 2,
      BLOCK / 2 - 2, 0, Math.PI * 2
    )
    ctx.fill()

    // Snake body
    game.snake.forEach((seg, i) => {
      if (i === 0) {
        // Head — brighter
        ctx.fillStyle = '#22c55e'
      } else {
        // Body — gradient darker toward tail
        const alpha = 1 - (i / game.snake.length) * 0.6
        ctx.fillStyle = `rgba(34, 197, 94, ${alpha})`
      }
      ctx.beginPath()
      ctx.roundRect(seg.x + 1, seg.y + 1, BLOCK - 2, BLOCK - 2, 4)
      ctx.fill()
    })

    // Game over overlay
    if (game.gameOver) {
      ctx.fillStyle = 'rgba(0,0,0,0.6)'
      ctx.fillRect(0, 0, width, height)
      ctx.fillStyle = '#ef4444'
      ctx.font = 'bold 32px monospace'
      ctx.textAlign = 'center'
      ctx.fillText('GAME OVER', width / 2, height / 2 - 10)
      ctx.fillStyle = '#ffffff'
      ctx.font = '18px monospace'
      ctx.fillText(`Score: ${game.score}`, width / 2, height / 2 + 24)
    }

  }, [game, width, height])

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      className="rounded-lg border border-gray-700"
    />
  )
}