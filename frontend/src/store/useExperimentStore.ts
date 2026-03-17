import { create } from 'zustand'
import type { Experiment, ProgressMessage } from '../types'

interface ExperimentStore {
  experiments: Experiment[]
  selectedId: string | null
  progress: Record<string, ProgressMessage>
  setExperiments: (experiments: Experiment[]) => void
  selectExperiment: (id: string | null) => void
  updateProgress: (msg: ProgressMessage) => void
  clearProgress: (experimentId: string) => void
}

export const useExperimentStore = create<ExperimentStore>((set) => ({
  experiments: [],
  selectedId: null,
  progress: {},

  setExperiments: (experiments) => set({ experiments }),

  selectExperiment: (id) => set({ selectedId: id }),

  updateProgress: (msg) =>
    set((state) => ({
      progress: { ...state.progress, [msg.experiment_id]: msg },
    })),

  clearProgress: (experimentId) =>
    set((state) => {
      const progress = { ...state.progress }
      delete progress[experimentId]
      return { progress }
    }),
}))
