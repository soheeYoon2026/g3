'use client'

import { useEffect, useRef, useState } from 'react'

import { Box, Flex, Grid, Image, Spinner, Stack, Text } from '@chakra-ui/react'
import { Lock, UploadSimple } from '@phosphor-icons/react/dist/ssr'

import { Button } from '@/components/ui/button'

import { G3ThreeViewer } from './g3-three-viewer'
import { EngineGlyph } from './project-engine-glyph'

type G3InferenceResult = {
  model: string
  elapsed_seconds: number
  grid: [number, number, number]
  drag_coefficient: number | null
  pressure_range: [number, number]
  speed_range: [number, number]
  device: string
  preview_png: string
}

const fixed = (value: number, digits = 2) =>
  Number.isFinite(value) ? value.toFixed(digits) : '—'

export function G3PreviewPanel() {
  const inputRef = useRef<HTMLInputElement>(null)
  const [stl, setStl] = useState<File | null>(null)
  const [stlUrl, setStlUrl] = useState<string>()
  const [result, setResult] = useState<G3InferenceResult>()
  const [error, setError] = useState<string>()
  const [isRunning, setIsRunning] = useState(false)

  useEffect(
    () => () => {
      if (stlUrl) URL.revokeObjectURL(stlUrl)
    },
    [stlUrl],
  )

  const selectStl = (file?: File) => {
    if (!file) return
    if (!file.name.toLowerCase().endsWith('.stl')) {
      setError('Please select an STL file.')
      return
    }
    setStl(file)
    setStlUrl(URL.createObjectURL(file))
    setResult(undefined)
    setError(undefined)
  }

  const runInference = async () => {
    if (!stl) {
      inputRef.current?.click()
      return
    }
    setIsRunning(true)
    setError(undefined)
    try {
      const body = new FormData()
      body.append('stl', stl)
      body.append('u_x', '30')
      body.append('ref_length', '5')
      body.append('ref_area', '3.380196')
      body.append('grid_x', '96')
      body.append('grid_y', '64')
      body.append('grid_z', '48')
      const response = await fetch('/api/bff/v1/g3/inferences', {
        method: 'POST',
        body,
      })
      const payload = (await response.json()) as G3InferenceResult & {
        detail?: string
      }
      if (!response.ok)
        throw new Error(payload.detail || 'G3 inference failed.')
      setResult(payload)
    } catch (caught) {
      setError(
        caught instanceof Error ? caught.message : 'G3 inference failed.',
      )
    } finally {
      setIsRunning(false)
    }
  }

  const metrics =
    result ?
      [
        {
          label: 'Drag coefficient (Cd)',
          value:
            result.drag_coefficient == null ?
              '—'
            : fixed(result.drag_coefficient, 4),
        },
        { label: 'Volume grid', value: result.grid.join(' × ') },
        {
          label: 'Pressure range',
          value: `${fixed(result.pressure_range[0])} ~ ${fixed(result.pressure_range[1])} Pa`,
        },
        {
          label: 'Speed range',
          value: `${fixed(result.speed_range[0])} ~ ${fixed(result.speed_range[1])} m/s`,
        },
        {
          label: 'Inference time',
          value: `${fixed(result.elapsed_seconds, 1)} s`,
        },
      ]
    : [
        { label: 'Drag coefficient (Cd)', value: 'Run inference' },
        { label: 'Volume grid', value: '96 × 64 × 48' },
        { label: 'Pressure range', value: '—' },
        { label: 'Speed range', value: '—' },
        { label: 'Inference time', value: '—' },
      ]

  return (
    <Stack gap="20px">
      <Flex align="center" justify="space-between" gap="16px" wrap="wrap">
        <Flex align="center" gap="12px">
          <EngineGlyph value="AIPhysics" size={42} />
          <Stack gap="2px">
            <Flex align="center" gap="8px">
              <Text color="grey.10" textStyle="pre-heading-2">
                G3 · AI + Physics
              </Text>
              <Flex
                align="center"
                gap="5px"
                px="8px"
                py="4px"
                borderRadius="full"
                bg="background.basic.2"
                color="grey.7"
              >
                <Lock size={11} weight="fill" />
                <Text textStyle="pre-caption-2" fontWeight="700">
                  PRE-BETA
                </Text>
              </Flex>
            </Flex>
            <Text color="grey.7" textStyle="pre-body-4">
              CFD-trained surrogate for pressure and three-dimensional flow.
            </Text>
          </Stack>
        </Flex>
      </Flex>

      <input
        ref={inputRef}
        hidden
        type="file"
        accept=".stl,model/stl"
        onChange={(event) => {
          selectStl(event.target.files?.[0])
          event.target.value = ''
        }}
      />

      <Flex
        p="16px"
        align="center"
        justify="space-between"
        gap="12px"
        wrap="wrap"
        borderWidth="1px"
        borderColor="border.basic.1"
        borderRadius="12px"
        bg="background.basic.1"
      >
        <Stack gap="2px" minW="0">
          <Text
            color="grey.10"
            textStyle="pre-body-4"
            fontWeight="700"
            truncate
          >
            {stl?.name ?? 'Select an STL for real-time inference'}
          </Text>
          <Text color={error ? 'red.500' : 'grey.6'} textStyle="pre-caption-2">
            {error ?? '30 m/s · 96 × 64 × 48 · admin-only GPU inference'}
          </Text>
        </Stack>
        <Flex gap="8px">
          <Button
            size="sm"
            color="grey"
            variant="outline"
            onClick={() => inputRef.current?.click()}
          >
            <UploadSimple size={16} />
            Select STL
          </Button>
          <Button
            size="sm"
            color="primary"
            loading={isRunning}
            loadingText="Inferring…"
            onClick={runInference}
          >
            Run G3
          </Button>
        </Flex>
      </Flex>

      <G3ThreeViewer stlUrl={stlUrl} />

      <Grid
        templateColumns={{
          base: '1fr',
          xl: 'minmax(0, 1.8fr) minmax(300px, 0.7fr)',
        }}
        gap="20px"
        alignItems="stretch"
      >
        <Box
          minH="520px"
          borderWidth="1px"
          borderColor="border.basic.1"
          borderRadius="16px"
          overflow="hidden"
          bg="#080A0C"
        >
          <Flex
            h="52px"
            px="18px"
            align="center"
            justify="space-between"
            borderBottomWidth="1px"
            borderBottomColor="rgba(255,255,255,0.10)"
          >
            <Text color="white" textStyle="pre-body-4" fontWeight="700">
              Surface pressure + 3D streamlines
            </Text>
            <Text color="rgba(255,255,255,0.58)" textStyle="pre-caption-2">
              {stl?.name ?? 'RS6 demo'} · 30 m/s
            </Text>
          </Flex>
          <Flex
            h="calc(100% - 52px)"
            minH="468px"
            align="center"
            justify="center"
          >
            {isRunning ?
              <Stack align="center" gap="12px">
                <Spinner color="primary.5" size="lg" />
                <Text color="rgba(255,255,255,0.7)" textStyle="pre-body-4">
                  Running CUDA field inference…
                </Text>
              </Stack>
            : result || !stl ?
              <Image
                src={
                  result?.preview_png ?? '/g3-preview/pressure-streamlines.png'
                }
                alt="G3 predicted surface pressure and streamlines"
                w="full"
                h="full"
                maxH="650px"
                objectFit="contain"
              />
            : <Text color="rgba(255,255,255,0.7)" textStyle="pre-body-4">
                Run G3 to generate this STL&apos;s pressure and streamline
                field.
              </Text>
            }
          </Flex>
        </Box>

        <Stack
          gap="14px"
          p="20px"
          borderWidth="1px"
          borderColor="border.basic.1"
          borderRadius="16px"
          bg="background.basic.1"
        >
          <Stack gap="4px">
            <Text color="grey.10" textStyle="pre-heading-4">
              Prediction summary
            </Text>
            <Text color="grey.6" textStyle="pre-caption-1">
              {result ?
                `${result.model} · ${result.device.toUpperCase()}`
              : 'Waiting for live inference'}
            </Text>
          </Stack>

          {metrics.map((metric) => (
            <Flex
              key={metric.label}
              align="center"
              justify="space-between"
              gap="12px"
              p="14px"
              borderRadius="10px"
              bg="background.basic.2"
            >
              <Text color="grey.6" textStyle="pre-caption-1">
                {metric.label}
              </Text>
              <Text color="grey.10" textStyle="pre-body-4" fontWeight="700">
                {metric.value}
              </Text>
            </Flex>
          ))}

          <Box mt="auto" p="14px" borderRadius="10px" bg="primary.1">
            <Text color="primary.7" textStyle="pre-caption-1" fontWeight="700">
              {result ? 'Live prediction complete' : 'Pre-beta inference'}
            </Text>
            <Text
              mt="4px"
              color="grey.7"
              textStyle="pre-caption-2"
              lineHeight="1.6"
            >
              {result ?
                'This pressure field and streamline image was generated from the uploaded STL.'
              : 'Select an STL and run the GPU model to replace the demonstration result.'
              }
            </Text>
          </Box>
        </Stack>
      </Grid>
    </Stack>
  )
}
