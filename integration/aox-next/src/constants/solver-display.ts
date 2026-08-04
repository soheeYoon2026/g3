export type SolverDisplayKey = 'fidelity' | 'fast' | 'frontier' | 'aiPhysics'

export const SOLVER_DISPLAY = {
  fidelity: {
    accentColor: '#03B403',
    badgeColor: 'green',
    chartColor: '#03C303',
    description: 'High-fidelity, highest accuracy',
    icon: '/images/engines/fidelity-icon.png',
    image: '/images/engines/fidelity.png',
    label: 'Fidelity',
    subtleBg: 'primary.1',
    subtleColor: 'primary.5',
    tagline: 'High fidelity',
  },
  fast: {
    accentColor: '#0C66E4',
    badgeColor: 'blue',
    chartColor: '#1D7AFC',
    description: 'Fast searching, rapid iteration',
    icon: '/images/engines/fast-icon.png',
    image: '/images/engines/fast.png',
    label: 'Fast',
    subtleBg: 'accent.blue1',
    subtleColor: 'accent.blue2',
    tagline: 'Fast searching',
  },
  frontier: {
    accentColor: '#C29428',
    badgeColor: 'yellow',
    chartColor: '#FBC037',
    description: 'New GPU-accelerated solver',
    icon: '/images/engines/frontier-icon.png',
    image: '/images/engines/frontier.png',
    label: 'Frontier',
    subtleBg: 'accent.yellow1',
    subtleColor: 'accent.yellow2',
    tagline: 'GPU solver',
  },
  aiPhysics: {
    accentColor: '#8B5CF6',
    badgeColor: 'purple',
    chartColor: '#A78BFA',
    description: 'AI surrogate grounded by CFD physics',
    icon: null,
    image: null,
    label: 'AI + Physics',
    subtleBg: 'background.basic.2',
    subtleColor: 'grey.9',
    tagline: 'Pre-beta',
  },
} as const

export const SOLVER_DISPLAY_ITEMS = [
  SOLVER_DISPLAY.fidelity,
  SOLVER_DISPLAY.fast,
  SOLVER_DISPLAY.frontier,
] as const

export function getSolverDisplayByMode(mode?: string | null) {
  const normalizedMode = mode?.trim().toUpperCase()

  if (normalizedMode === 'G1' || normalizedMode === 'FIDELITY') {
    return SOLVER_DISPLAY.fidelity
  }

  if (normalizedMode === 'G2' || normalizedMode === 'FAST') {
    return SOLVER_DISPLAY.fast
  }

  if (normalizedMode === 'G4' || normalizedMode === 'FRONTIER') {
    return SOLVER_DISPLAY.frontier
  }

  if (normalizedMode === 'G3' || normalizedMode === 'AIPHYSICS') {
    return SOLVER_DISPLAY.aiPhysics
  }

  return null
}
