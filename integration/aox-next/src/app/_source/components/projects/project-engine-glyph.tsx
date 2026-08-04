'use client'

import { useState } from 'react'

import { Box, Flex, Image } from '@chakra-ui/react'
import {
  Atom,
  GraphicsCard,
  Lightning,
  Target,
} from '@phosphor-icons/react/dist/ssr'

import { getSolverDisplayByMode } from '@/constants/solver-display'

export type ProjectEngineType = 'Fidelity' | 'Fast' | 'Frontier' | 'AIPhysics'

const ENGINE_ICON: Record<ProjectEngineType, typeof Target> = {
  Fidelity: Target,
  Fast: Lightning,
  Frontier: GraphicsCard,
  AIPhysics: Atom,
}

export function EngineGlyph({
  value,
  size = 22,
}: {
  value: ProjectEngineType
  size?: number
}) {
  const display = getSolverDisplayByMode(value)
  const Icon = ENGINE_ICON[value]
  const accent = display?.accentColor ?? '#6A6D71'
  const iconSrc = display?.icon
  const radius = `${Math.round(size * 0.28)}px`
  const [imgFailed, setImgFailed] = useState(false)

  if (iconSrc && !imgFailed) {
    return (
      <Box
        boxSize={`${size}px`}
        borderRadius={radius}
        overflow="hidden"
        flexShrink={0}
      >
        <Image
          src={iconSrc}
          alt={`${display?.label ?? value} engine`}
          w="full"
          h="full"
          objectFit="cover"
          onError={() => setImgFailed(true)}
        />
      </Box>
    )
  }

  return (
    <Flex
      align="center"
      justify="center"
      boxSize={`${size}px`}
      borderRadius={radius}
      bg={`${accent}1A`}
      flexShrink={0}
    >
      <Icon size={Math.round(size * 0.6)} color={accent} weight="fill" />
    </Flex>
  )
}
