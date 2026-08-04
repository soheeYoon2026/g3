'use client'

import type { ReactNode } from 'react'

import { Box, Center, Flex, Spinner, Stack, Text } from '@chakra-ui/react'
import { type ArrowLeft, Lock } from '@phosphor-icons/react/dist/ssr'

import { getSolverDisplayByMode } from '@/constants/solver-display'
import type { useInfiniteScroll } from '@/hooks/use-infinite-scroll'

import { EngineGlyph, type ProjectEngineType } from './project-engine-glyph'

type ProjectFilterEngine = Exclude<ProjectEngineType, 'AIPhysics'>
const projectTypes: ProjectFilterEngine[] = ['Fidelity', 'Fast', 'Frontier']
const onboardingEngineSelectedEvent = 'aox:onboarding-engine-selected'

export function ProjectHomeContextSidebar({
  collapsed,
  isFetchingNextPage,
  isLoading,
  loadMoreRef,
  onFilterChange,
  onG3Click,
  projectsCount,
  selectedFilter,
  g3Active = false,
  showG3 = false,
}: {
  collapsed: boolean
  isFetchingNextPage: boolean
  isLoading: boolean
  loadMoreRef: ReturnType<typeof useInfiniteScroll>
  onFilterChange: (filter: 'all' | ProjectFilterEngine) => void
  onG3Click?: () => void
  projectsCount: number
  selectedFilter: 'all' | ProjectFilterEngine
  g3Active?: boolean
  showG3?: boolean
}) {
  const solverFilters = projectTypes.map((value) => ({
    label: `X-${value}`,
    value,
  }))

  return (
    <>
      <SidebarFilterButton
        active={selectedFilter === 'all'}
        collapsed={collapsed}
        label="All projects"
        onClick={() => onFilterChange('all')}
      />

      <Box data-onboarding={!collapsed ? 'project-engine-options' : undefined}>
        {!collapsed && (
          <Text
            mt="14px"
            mb="4px"
            px="10px"
            color="grey.5"
            textStyle="pre-caption-2"
            fontWeight="700"
            letterSpacing="0.6px"
            textTransform="uppercase"
          >
            Engine
          </Text>
        )}

        {solverFilters.map((filter) => {
          const display = getSolverDisplayByMode(filter.value)
          return (
            <SidebarFilterButton
              key={filter.value}
              active={selectedFilter === filter.value}
              collapsed={collapsed}
              description={display?.tagline}
              iconNode={
                <EngineGlyph value={filter.value} size={collapsed ? 24 : 26} />
              }
              label={filter.label}
              onboardingEngineOption
              onClick={() => {
                onFilterChange(filter.value)
                window.dispatchEvent(new Event(onboardingEngineSelectedEvent))
              }}
            />
          )
        })}

        {showG3 && (
          <SidebarFilterButton
            active={g3Active}
            collapsed={collapsed}
            description="AI + Physics · Pre-beta"
            iconNode={
              <EngineGlyph value="AIPhysics" size={collapsed ? 24 : 26} />
            }
            label="G3 (AI + Physics)"
            locked
            onClick={() => onG3Click?.()}
          />
        )}
      </Box>

      {!collapsed && (
        <Box
          mt="14px"
          p="12px"
          borderWidth="1px"
          borderColor="border.basic.1"
          borderRadius="10px"
          bg="background.basic.2"
        >
          <Text color="grey.6" textStyle="pre-caption-1">
            {selectedFilter === 'all' ?
              'Projects'
            : `${selectedFilter} projects`}
          </Text>
          <Text color="grey.10" textStyle="pre-heading-4">
            {isLoading ? '-' : projectsCount}
          </Text>
        </Box>
      )}

      <Box ref={loadMoreRef} h="1px" />
      {isFetchingNextPage ?
        <Center py="12px">
          <Spinner color="grey.7" />
        </Center>
      : null}
    </>
  )
}

function SidebarFilterButton({
  active,
  collapsed,
  description,
  dotColor,
  iconNode,
  label,
  leadingIcon: LeadingIcon,
  locked = false,
  onboardingEngineOption,
  onClick,
}: {
  active: boolean
  collapsed: boolean
  description?: string
  dotColor?: string
  iconNode?: ReactNode
  label: string
  leadingIcon?: typeof ArrowLeft
  locked?: boolean
  onboardingEngineOption?: boolean
  onClick: () => void
}) {
  return (
    <Flex
      as="button"
      title={collapsed ? label : undefined}
      w={collapsed ? '36px' : 'full'}
      minH="36px"
      px={collapsed ? '0' : '10px'}
      py="7px"
      align="center"
      justify={collapsed ? 'center' : 'flex-start'}
      gap="10px"
      borderRadius="8px"
      bg={active ? 'primary.1' : 'transparent'}
      textAlign="left"
      cursor="pointer"
      position="relative"
      data-onboarding-engine-option={
        onboardingEngineOption ? 'true' : undefined
      }
      onClick={onClick}
    >
      {iconNode ?
        iconNode
      : collapsed ?
        <Box
          w={
            dotColor ? '8px'
            : active ?
              '18px'
            : '8px'
          }
          h="8px"
          borderRadius="full"
          bg={dotColor ?? (active ? 'primary.5' : 'grey.5')}
          opacity={dotColor && !active ? 0.55 : 1}
          transition="all 0.16s ease"
        />
      : LeadingIcon ?
        <LeadingIcon
          size={16}
          color={active ? '#03B403' : '#6A6D71'}
          weight="bold"
        />
      : <Box
          boxSize="8px"
          borderRadius="full"
          bg={dotColor ?? (active ? 'primary.5' : 'grey.5')}
          opacity={dotColor && !active ? 0.55 : 1}
          flexShrink={0}
        />
      }
      {!collapsed && (
        <Stack gap="1px" minW="0">
          <Text
            color={active ? 'primary.6' : 'grey.8'}
            textStyle="pre-caption-1"
            fontWeight={active ? '600' : '500'}
            overflow="hidden"
            textOverflow="ellipsis"
            whiteSpace="nowrap"
          >
            {label}
          </Text>
          {description ?
            <Text
              color="grey.5"
              textStyle="pre-caption-2"
              overflow="hidden"
              textOverflow="ellipsis"
              whiteSpace="nowrap"
            >
              {description}
            </Text>
          : null}
        </Stack>
      )}
      {locked && (
        <Center
          ml={collapsed ? '0' : 'auto'}
          position={collapsed ? 'absolute' : 'static'}
          top={collapsed ? '2px' : undefined}
          right={collapsed ? '2px' : undefined}
          boxSize={collapsed ? '13px' : '20px'}
          borderRadius="full"
          bg={active ? 'primary.2' : 'background.basic.2'}
          color={active ? 'primary.6' : 'grey.6'}
          flexShrink={0}
        >
          <Lock size={collapsed ? 8 : 12} weight="fill" />
        </Center>
      )}
    </Flex>
  )
}
