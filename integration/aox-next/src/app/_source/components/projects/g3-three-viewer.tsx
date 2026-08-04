'use client'

/* eslint-disable react/no-unknown-property --
   R3F(@react-three/fiber) 는 mesh/light 등에 geometry·castShadow·intensity
   같은 three.js prop 을 넘긴다. eslint react 플러그인은 이를 DOM 속성으로
   오해해 no-unknown-property 를 낸다 — morph-model-viewer.tsx 와 동일 예외. */
import { Suspense } from 'react'

import { Box, Flex, Text } from '@chakra-ui/react'
import { Bounds, Center, OrbitControls } from '@react-three/drei'
import { Canvas, useLoader } from '@react-three/fiber'

import type { BufferGeometry } from 'three'
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader'

function StlMesh({ url }: { url: string }) {
  // useLoader의 STLLoader 반환 타입이 unknown 으로 좁혀져 mesh geometry prop 과
  // 안 맞음 — BufferGeometry 로 명시 캐스팅.
  const geometry = useLoader(
    STLLoader,
    url,
  ) as unknown as BufferGeometry

  return (
    <Bounds fit clip observe margin={1.15}>
      <Center>
        <mesh geometry={geometry} rotation={[-Math.PI / 2, 0, 0]} castShadow>
          <meshStandardMaterial
            color="#A78BFA"
            metalness={0.18}
            roughness={0.48}
          />
        </mesh>
      </Center>
    </Bounds>
  )
}

export function G3ThreeViewer({ stlUrl }: { stlUrl?: string }) {
  return (
    <Box
      h={{ base: '320px', md: '400px' }}
      borderWidth="1px"
      borderColor="border.basic.1"
      borderRadius="16px"
      overflow="hidden"
      bg="#111318"
      position="relative"
    >
      <Flex
        position="absolute"
        zIndex="1"
        top="14px"
        left="16px"
        px="10px"
        py="6px"
        borderRadius="8px"
        bg="rgba(8,10,12,0.72)"
        pointerEvents="none"
      >
        <Text color="white" textStyle="pre-caption-1" fontWeight="700">
          Three.js geometry preview · drag to orbit · wheel to zoom
        </Text>
      </Flex>

      <Canvas camera={{ position: [6, 3.5, 5], fov: 34 }} shadows>
        <color attach="background" args={['#111318']} />
        <ambientLight intensity={1.4} />
        <directionalLight position={[5, 8, 6]} intensity={3.2} castShadow />
        <directionalLight position={[-4, 2, -5]} intensity={1.2} />
        <Suspense fallback={null}>
          <StlMesh url={stlUrl ?? '/g3-preview/rs6-preview.stl'} />
        </Suspense>
        <OrbitControls makeDefault enableDamping dampingFactor={0.08} />
      </Canvas>
    </Box>
  )
}
