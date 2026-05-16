interface Props {
  running: boolean
  onStart: () => void
  onStop: () => void
  onReset: () => void
}

export default function Controls({ running, onStart, onStop, onReset }: Props) {
  return (
    <div className="flex gap-3 mt-4">
      <button
        onClick={onStart}
        disabled={running}
        className="flex-1 py-2 px-4 rounded-lg font-semibold text-sm
          bg-green-600 hover:bg-green-500 disabled:bg-gray-700
          disabled:text-gray-500 text-white transition-colors duration-150"
      >
        ▶ Start
      </button>

      <button
        onClick={onStop}
        disabled={!running}
        className="flex-1 py-2 px-4 rounded-lg font-semibold text-sm
          bg-yellow-600 hover:bg-yellow-500 disabled:bg-gray-700
          disabled:text-gray-500 text-white transition-colors duration-150"
      >
        ⏸ Pause
      </button>

      <button
        onClick={onReset}
        className="flex-1 py-2 px-4 rounded-lg font-semibold text-sm
          bg-red-700 hover:bg-red-600 text-white transition-colors duration-150"
      >
        ↺ Reset
      </button>
    </div>
  )
}