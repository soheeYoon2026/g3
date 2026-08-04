'use client'

import { useMemo, useState } from 'react'

import { useRouter } from 'next/navigation'

import { Box, Flex, Stack, Text } from '@chakra-ui/react'
import { CaretLeft, CaretRight } from '@phosphor-icons/react/dist/ssr'
import { useQueryClient } from '@tanstack/react-query'

import { useTeamContext } from '@/app/_source/components/job/team-context-provider'
import { toaster } from '@/components/ui/toaster'
import {
  QUERY_KEY_PROJECTS_API,
  useProjectsDestroyMutation,
  useProjectsListInfiniteQuery,
  useProjectsUpdateMutation,
} from '@/generated/apis/Projects/Projects.query'
import { useInfiniteScroll } from '@/hooks/use-infinite-scroll'

import { G3PreviewPanel } from './g3-preview-panel'
import { ProjectHomeContextSidebar } from './project-home-context-sidebar'
import { ProjectsHomePanel } from './project-home-panel'
import { toProjectsProject } from './project-page-utils'

type ProjectModeLabelFilter = 'Fidelity' | 'Fast' | 'Frontier'
const projectModeByLabelFilter: Record<
  ProjectModeLabelFilter,
  'G1' | 'G2' | 'G4'
> = {
  Fidelity: 'G1',
  Fast: 'G2',
  Frontier: 'G4',
}

function getProjectPath(projectUid: string) {
  return `/project/${encodeURIComponent(projectUid)}`
}

function getProjectWorkbenchPath(projectUid: string) {
  return `${getProjectPath(projectUid)}/workbench?from=projects`
}

function getProjectRunsPath(projectUid: string) {
  return `${getProjectPath(projectUid)}/runs`
}

export const ProjectsPage = () => {
  const router = useRouter()
  const queryClient = useQueryClient()
  const { selectedTeam } = useTeamContext()
  const canAccessG3 = ['team_owner', 'team_admin'].includes(
    selectedTeam?.role?.toLowerCase() ?? '',
  )
  const [selectedEngineFilter, setSelectedEngineFilter] =
    useState<ProjectModeLabelFilter | null>(null)
  const [showG3Preview, setShowG3Preview] = useState(false)
  const selectedProjectMode =
    selectedEngineFilter == null ? null : (
      projectModeByLabelFilter[selectedEngineFilter]
    )
  const [projectSidebarCollapsed, setProjectSidebarCollapsed] = useState(false)
  const [projectFilter, setProjectFilter] = useState<
    'all' | ProjectModeLabelFilter
  >('all')
  const projectListMode =
    projectFilter === 'all' ? undefined : (
      projectModeByLabelFilter[projectFilter]
    )
  const { data, fetchNextPage, hasNextPage, isFetchingNextPage, isLoading } =
    useProjectsListInfiniteQuery({
      variables: {
        query: {
          page_size: 20,
          ...(projectListMode ? { mode: projectListMode } : {}),
        },
      },
      options: {
        staleTime: 0,
      },
    })
  const loadMoreRef = useInfiniteScroll({
    fetchNextPage,
    hasNextPage,
    isFetchingNextPage,
  })
  const { mutateAsync: deleteProject } = useProjectsDestroyMutation({
    options: {},
  })
  const { mutateAsync: updateProject } = useProjectsUpdateMutation({
    options: {},
  })
  const projects = useMemo(
    () => data?.pages.flatMap((page) => page.results ?? []) ?? [],
    [data?.pages],
  )
  const workspaceProjects = useMemo(
    () => projects.map(toProjectsProject),
    [projects],
  )
  const totalProjectsCount = data?.pages[0]?.count ?? workspaceProjects.length

  const handleProjectFilterChange = (
    filter: 'all' | ProjectModeLabelFilter,
  ) => {
    setShowG3Preview(false)
    setProjectFilter(filter)

    const nextType = filter === 'all' ? null : filter

    setSelectedEngineFilter(nextType)
  }

  const handleProjectSelect = (projectUid: string) => {
    router.push(getProjectRunsPath(projectUid), {
      scroll: false,
    })
  }

  const handleProjectRename = async (
    project: (typeof workspaceProjects)[number],
    title: string,
  ) => {
    try {
      await updateProject({
        id: project.id,
        data: {
          name: title,
          mode: project.requestMode ?? selectedProjectMode ?? 'G1',
          description: project.description,
          archivedAt: project.archivedAt,
        },
      })

      await queryClient.invalidateQueries({
        queryKey: QUERY_KEY_PROJECTS_API.LIST(),
      })
      await queryClient.invalidateQueries({
        queryKey: QUERY_KEY_PROJECTS_API.LIST_INFINITE(),
      })

      toaster.create({
        type: 'success',
        title: 'Project renamed.',
      })
    } catch (error) {
      toaster.create({
        type: 'error',
        title: 'Failed to rename project.',
      })

      throw error
    }
  }

  const handleProjectDelete = async (
    project: (typeof workspaceProjects)[number],
  ) => {
    try {
      await deleteProject({ id: project.id })

      await queryClient.invalidateQueries({
        queryKey: QUERY_KEY_PROJECTS_API.LIST(),
      })
      await queryClient.invalidateQueries({
        queryKey: QUERY_KEY_PROJECTS_API.LIST_INFINITE(),
      })

      toaster.create({
        type: 'success',
        title: 'Project deleted.',
      })
    } catch (error) {
      toaster.create({
        type: 'error',
        title: 'Failed to delete project.',
      })

      throw error
    }
  }

  return (
    <Flex h="100%" minH="0" align="stretch" overflow="hidden">
      <Box
        w={projectSidebarCollapsed ? '56px' : '220px'}
        minW={projectSidebarCollapsed ? '56px' : '220px'}
        h="100%"
        minH="0"
        flexShrink={0}
        bg="background.basic.1"
        borderRightWidth="1px"
        borderRightColor="border.basic.1"
        overflow="hidden"
        transition="width 0.2s ease, min-width 0.2s ease"
      >
        <Flex
          h="76px"
          px={projectSidebarCollapsed ? '8px' : '14px'}
          align="center"
          justify={projectSidebarCollapsed ? 'center' : 'space-between'}
          gap="8px"
          borderBottomWidth="1px"
          borderBottomColor="border.basic.1"
        >
          {!projectSidebarCollapsed && (
            <Text color="grey.8" textStyle="pre-body-3" fontWeight="700">
              PROJECTS
            </Text>
          )}
          <Flex
            as="button"
            align="center"
            justify="center"
            minW="26px"
            h="26px"
            p="0"
            borderRadius="7px"
            bg="transparent"
            color="grey.6"
            cursor="pointer"
            transition="all 0.12s ease"
            _hover={{ bg: 'background.basic.2', color: 'grey.9' }}
            aria-label={
              projectSidebarCollapsed ?
                'Expand project sidebar'
              : 'Collapse project sidebar'
            }
            onClick={() => setProjectSidebarCollapsed((current) => !current)}
          >
            {projectSidebarCollapsed ?
              <CaretRight size={15} />
            : <CaretLeft size={15} />}
          </Flex>
        </Flex>

        <Stack
          h="calc(100% - 76px)"
          p={projectSidebarCollapsed ? '10px 6px' : '12px 10px'}
          gap="5px"
          overflowY="auto"
        >
          <ProjectHomeContextSidebar
            collapsed={projectSidebarCollapsed}
            isFetchingNextPage={isFetchingNextPage}
            isLoading={isLoading}
            loadMoreRef={loadMoreRef}
            projectsCount={totalProjectsCount}
            selectedFilter={projectFilter}
            onFilterChange={handleProjectFilterChange}
            g3Active={showG3Preview}
            showG3={canAccessG3}
            onG3Click={() => {
              setShowG3Preview(true)
              setSelectedEngineFilter(null)
              setProjectFilter('all')
            }}
          />
        </Stack>
      </Box>

      <Flex flex="1" minW="0" h="100%" minH="0" direction="column">
        <Box flex="1" minW="0" h="100%" minH="0" overflow="auto" p="48px">
          {showG3Preview && canAccessG3 ?
            <G3PreviewPanel />
          : <ProjectsHomePanel
              selectedType={selectedEngineFilter}
              selectedProjectMode={selectedProjectMode}
              filteredProjects={workspaceProjects}
              isFetchingNextPage={isFetchingNextPage}
              isLoading={isLoading}
              loadMoreRef={loadMoreRef}
              onDelete={handleProjectDelete}
              onRename={handleProjectRename}
              onSelect={handleProjectSelect}
              onWorkbenchCreated={(project) => {
                router.push(getProjectWorkbenchPath(project.uid), {
                  scroll: false,
                })
              }}
            />
          }
        </Box>
      </Flex>
    </Flex>
  )
}
