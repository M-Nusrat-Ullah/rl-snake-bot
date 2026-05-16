interface Props {
  score: number
  bestScore: number
  episode: number
  qValues: number[]
}

const ACTION_LABELS = ['Straight', 'Right', 'Left']
const ACTION_COLORS = ['bg-blue-500', 'bg-yellow-500', 'bg-purple-500']

export default function StatsPanel({ score, bestScore, episode, qValues }: Props) {
  const maxQ   = Math.max(...qValues)
  const minQ   = Math.min(...qValues)
  const rangeQ = maxQ - minQ || 1

  return (
    <div className="flex flex-col gap-4 w-52">

      {/* Scores */}
      <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
        <h2 className="text-gray-400 text-xs uppercase tracking-widest mb-3">
          Stats
        </h2>
        <div className="flex flex-col gap-2">
          <div className="flex justify-between items-center">
            <span className="text-gray-400 text-sm">Episode</span>
            <span className="text-white font-mono font-bold">{episode}</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-gray-400 text-sm">Score</span>
            <span className="text-green-400 font-mono font-bold text-lg">
              {score}
            </span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-gray-400 text-sm">Best</span>
            <span className="text-yellow-400 font-mono font-bold text-lg">
              {bestScore}
            </span>
          </div>
        </div>
      </div>

      {/* Q-Values */}
      <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
        <h2 className="text-gray-400 text-xs uppercase tracking-widest mb-3">
          Q-Values
        </h2>
        <div className="flex flex-col gap-2">
          {qValues.map((q, i) => {
            const pct = ((q - minQ) / rangeQ) * 100
            const isChosen = q === maxQ
            return (
              <div key={i}>
                <div className="flex justify-between text-xs mb-1">
                  <span className={isChosen ? 'text-white font-bold' : 'text-gray-400'}>
                    {isChosen ? '▶ ' : ''}{ACTION_LABELS[i]}
                  </span>
                  <span className="text-gray-300 font-mono">
                    {q.toFixed(2)}
                  </span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-1.5">
                  <div
                    className={`${ACTION_COLORS[i]} h-1.5 rounded-full transition-all duration-150`}
                    style={{ width: `${pct}%` }}
                  />
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {/* Legend */}
      <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
        <h2 className="text-gray-400 text-xs uppercase tracking-widest mb-3">
          Legend
        </h2>
        <div className="flex flex-col gap-2 text-xs">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-sm bg-green-500"/>
            <span className="text-gray-300">Snake</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-red-500"/>
            <span className="text-gray-300">Food</span>
          </div>
        </div>
      </div>

    </div>
  )
}