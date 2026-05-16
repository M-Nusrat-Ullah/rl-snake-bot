import SnakeCanvas from './components/SnakeCanvas'
import StatsPanel from './components/StatsPanel'
import Controls from './components/Controls'
import { useSnakeGame } from './hooks/useSnakeGame'

export default function App() {
  const { game, running, episode, bestScore, qValues, start, stop, reset } =
    useSnakeGame()

  return (
    <div className="min-h-screen bg-gray-900 text-white flex flex-col items-center justify-center p-8">

      {/* Header */}
      <div className="mb-6 text-center">
        <h1 className="text-3xl font-bold tracking-tight">
          🐍 RL Snake Bot
        </h1>
        <p className="text-gray-400 text-sm mt-1">
          Deep Q-Network agent playing Snake in real time
        </p>
      </div>

      {/* Main layout */}
      <div className="flex gap-6 items-start">

        {/* Left — canvas + controls */}
        <div className="flex flex-col items-center">
          <SnakeCanvas game={game} />
          <Controls
            running={running}
            onStart={start}
            onStop={stop}
            onReset={reset}
          />
        </div>

        {/* Right — stats */}
        <StatsPanel
          score={game.score}
          bestScore={bestScore}
          episode={episode}
          qValues={qValues}
        />

      </div>

      {/* Footer */}
      <p className="mt-8 text-gray-600 text-xs">
        Built by M. Nusrat Ullah · DQN + PyTorch + FastAPI + React
      </p>

    </div>
  )
}