import { RigidBody } from "@react-three/rapier";
import { useFrame } from "@react-three/fiber";
import { useMemo, useRef } from "react";
import { createNoise2D } from "simplex-noise";
import * as THREE from "three";
import ClaudeGrassQuick3 from "./ClaudeGrassQuick3";
import { GrassField } from "./GrassClaude7";
import { useGrassClaude7Controls } from "./useGrassClaude7Controls";

export const HillySideScrollerMap = ({
  playerPosition = [0, 0, 0],
  grassControls,
  ...props
}) => {
  // Get GrassClaude7 controls
  const grassClaude7Controls = useGrassClaude7Controls();
  const groundRef = useRef();
  const playerPosRef = useRef(new THREE.Vector3(...playerPosition));

  // Update player position reactively
  useFrame(() => {
    if (Array.isArray(playerPosition) && playerPosition.length >= 3) {
      playerPosRef.current.set(
        playerPosition[0],
        playerPosition[1],
        playerPosition[2]
      );
    }
  });

  // Generate 500x500 procedural terrain with hills
  // Hills are flatter near character (Z=0) and more prominent in background (negative Z)
  const { terrainGeometry, terrainMaterial, heightData, heightmapTexture } = useMemo(() => {
    const size = 500; // 500x500 terrain
    const segments = 200; // High detail for smooth hills
    const maxHeight = 15; // Maximum hill height (slightly increased for bigger hills)
    const noiseScale = 0.005; // Lower scale for bigger, smoother hills (fewer, larger features)

    // Create simplex noise
    const noise2D = createNoise2D(() => Math.random());

    // Create plane geometry
    const geometry = new THREE.PlaneGeometry(size, size, segments, segments);
    const positions = geometry.attributes.position;
    const vertexCount = segments + 1;
    const heightData = new Float32Array((segments + 1) * (segments + 1));

    // Generate smooth hills using noise
    for (let i = 0; i < positions.count; i++) {
      const x = i % vertexCount;
      const z = Math.floor(i / vertexCount);

      // Normalize to 0-1 range
      const nx = x / segments;
      const nz = z / segments;

      // Map to world coordinates (centered at origin)
      // X: -250 to 250 (left-right)
      // Z: -250 to 250 (back to front, where character is at Z=0)
      const worldX = (nx - 0.5) * size;
      const worldZ = (nz - 0.5) * size;

      // Use multiple octaves for smooth hills
      let height = 0;
      let amplitude = 1;
      let frequency = 1;
      let maxValue = 0;

      // 2 octaves for very smooth, large hills (fewer octaves = smoother, less detail)
      for (let octave = 0; octave < 2; octave++) {
        const sampleX = worldX * frequency * noiseScale;
        const sampleZ = worldZ * frequency * noiseScale;
        height += noise2D(sampleX, sampleZ) * amplitude;
        maxValue += amplitude;
        amplitude *= 0.5;
        frequency *= 2;
      }

      // Normalize to 0-1 range
      height = (height / maxValue + 1) * 0.5;

      // Apply very smooth curve for gentle, rolling hills
      height = Math.pow(height, 1.1);

      // Scale to desired height
      height *= maxHeight;

      // For 2.5D side view: hills should be flatter near character (Z=0) 
      // and more prominent in background (negative Z, further from camera)
      // Character is at Z=0, background is negative Z
      
      // Keep it very flat near character (Z=0 area)
      const flatZone = 15; // Keep flat within 15 units of Z=0 (very close hills)
      if (Math.abs(worldZ) < flatZone) {
        // Very flat near character - fade from 0 to small height
        const flatFactor = Math.abs(worldZ) / flatZone; // 0 at center, 1 at edge
        height *= flatFactor * 0.15; // Very flat near character
      } else if (worldZ < 0) {
        // Background (negative Z) - hills increase as we go further back
        const backDistance = Math.abs(worldZ) - flatZone; // Distance beyond flat zone
        const maxBackDistance = size / 2 - flatZone; // Maximum distance to edge
        const normalizedBack = Math.min(backDistance / maxBackDistance, 1);
        // Hills increase more quickly toward background (lower exponent = faster increase)
        const fadeFactor = Math.pow(normalizedBack, 0.5);
        height *= 0.15 + fadeFactor * 0.85; // Start from flat zone height, increase to full
      } else {
        // Foreground (positive Z) - keep relatively flat
        const frontDistance = worldZ - flatZone;
        const maxFrontDistance = size / 2 - flatZone;
        const normalizedFront = Math.min(frontDistance / maxFrontDistance, 1);
        const fadeFactor = Math.pow(normalizedFront, 0.8);
        height *= 0.15 + fadeFactor * 0.4; // Less hills in foreground
      }

      // Store height data for potential use
      const heightIndex = z * vertexCount + x;
      heightData[heightIndex] = height;

      // Set the Y coordinate (height) - PlaneGeometry uses Z as height before rotation
      positions.setZ(i, height);
    }

    // Recalculate normals for proper lighting
    geometry.computeVertexNormals();

    // Create material with a nice terrain color
    const material = new THREE.MeshStandardMaterial({
      color: 0x6b8e23, // Olive green terrain color
      roughness: 0.9,
      metalness: 0.0,
    });

    // Create heightmap texture from heightData for grass
    // Use 512x512 resolution for better detail and precision
    const heightmapResolution = 512;
    const textureData = new Float32Array(heightmapResolution * heightmapResolution);
    
    // Fill texture data by sampling heightData with bilinear interpolation
    // Normalize height to 0-1 range based on maxHeight (not min/max of actual data)
    // This ensures the shader can correctly scale it back using terrainHeight
    for (let y = 0; y < heightmapResolution; y++) {
      for (let x = 0; x < heightmapResolution; x++) {
        // Map texture coordinates to heightData indices
        const u = x / (heightmapResolution - 1);
        const v = y / (heightmapResolution - 1);
        
        const dataX = u * segments;
        const dataY = v * segments;
        
        const x0 = Math.floor(dataX);
        const y0 = Math.floor(dataY);
        const x1 = Math.min(x0 + 1, segments);
        const y1 = Math.min(y0 + 1, segments);
        
        // Bilinear interpolation
        const fx = dataX - x0;
        const fy = dataY - y0;
        
        const h00 = heightData[y0 * (segments + 1) + x0] / maxHeight;
        const h10 = heightData[y0 * (segments + 1) + x1] / maxHeight;
        const h01 = heightData[y1 * (segments + 1) + x0] / maxHeight;
        const h11 = heightData[y1 * (segments + 1) + x1] / maxHeight;
        
        const h0 = h00 * (1 - fx) + h10 * fx;
        const h1 = h01 * (1 - fx) + h11 * fx;
        const height = h0 * (1 - fy) + h1 * fy;
        
        textureData[y * heightmapResolution + x] = Math.max(0, Math.min(1, height));
      }
    }
    
    // Create DataTexture (R channel only for height)
    const heightmapTexture = new THREE.DataTexture(
      textureData,
      heightmapResolution,
      heightmapResolution,
      THREE.RedFormat,
      THREE.FloatType
    );
    heightmapTexture.needsUpdate = true;
    heightmapTexture.wrapS = THREE.ClampToEdgeWrapping;
    heightmapTexture.wrapT = THREE.ClampToEdgeWrapping;
    heightmapTexture.minFilter = THREE.LinearFilter;
    heightmapTexture.magFilter = THREE.LinearFilter;

    return { terrainGeometry: geometry, terrainMaterial: material, heightData, heightmapTexture };
  }, []);

  return (
    <group {...props}>
      {/* 500x500 Procedural terrain with hills */}
      {/* Hills are flatter near character (Z=0) and more prominent in background */}
      <RigidBody type="fixed" colliders="trimesh">
        <mesh
          ref={groundRef}
          geometry={terrainGeometry}
          material={terrainMaterial}
          rotation={[-Math.PI / 2, 0, 0]}
          position={[0, 0, 0]}
          receiveShadow
          castShadow
        />
      </RigidBody>

      {/* Some decorative elements - trees or rocks on hills */}
      {useMemo(() => {
        // Function to sample height from heightmap at given world coordinates
        const getHeightAt = (worldX, worldZ) => {
          const size = 500;
          const segments = 200;
          const vertexCount = segments + 1;

          // Convert world coordinates to normalized coordinates (0-1)
          const nx = (worldX + size / 2) / size; // -250 to 250 -> 0 to 1
          const nz = (worldZ + size / 2) / size; // -250 to 250 -> 0 to 1

          // Clamp to valid range
          const clampedX = Math.max(0, Math.min(1, nx));
          const clampedZ = Math.max(0, Math.min(1, nz));

          // Convert to array indices
          const xIndex = Math.floor(clampedX * segments);
          const zIndex = Math.floor(clampedZ * segments);

          // Get the four surrounding vertices for bilinear interpolation
          const x0 = Math.max(0, Math.min(segments, xIndex));
          const x1 = Math.max(0, Math.min(segments, xIndex + 1));
          const z0 = Math.max(0, Math.min(segments, zIndex));
          const z1 = Math.max(0, Math.min(segments, zIndex + 1));

          // Get heights at the four corners
          const h00 = heightData[z0 * vertexCount + x0];
          const h10 = heightData[z0 * vertexCount + x1];
          const h01 = heightData[z1 * vertexCount + x0];
          const h11 = heightData[z1 * vertexCount + x1];

          // Bilinear interpolation
          const fx = (clampedX * segments) - xIndex;
          const fz = (clampedZ * segments) - zIndex;

          const h0 = h00 * (1 - fx) + h10 * fx;
          const h1 = h01 * (1 - fx) + h11 * fx;
          const height = h0 * (1 - fz) + h1 * fz;

          return height;
        };

        // Tree positions in world coordinates (X, Z)
        // Y will be calculated from heightmap
        // Trees positioned very close to character
        const treePositions = [
          { x: -60, z: -20 },
          { x: -50, z: -25 },
          { x: -40, z: -22 },
          { x: -30, z: -28 },
          { x: -20, z: -24 },
          { x: 20, z: -26 },
          { x: 30, z: -30 },
          { x: 40, z: -25 },
          { x: 50, z: -29 },
          { x: 60, z: -24 },
          // More trees scattered in background (very close)
          { x: -80, z: -35 },
          { x: -70, z: -38 },
          { x: 70, z: -34 },
          { x: 80, z: -36 },
          { x: -90, z: -40 },
          { x: 90, z: -42 },
        ];

        return treePositions.map((pos, i) => {
          // Calculate height from heightmap
          const terrainHeight = getHeightAt(pos.x, pos.z);
          // CylinderGeometry is centered, so trunk goes from -1 to +1 in local Y
          // We want the bottom of the trunk at terrainHeight, so offset by +1
          const trunkBottomOffset = 1; // Half of trunk height (2/2 = 1)
          const treeY = terrainHeight + trunkBottomOffset;

          return (
            <RigidBody
              key={`decoration-${i}`}
              type="fixed"
              colliders="cuboid"
              position={[pos.x, treeY, pos.z]}
            >
              <mesh castShadow receiveShadow>
                {/* Simple tree representation */}
                <group>
                  {/* Trunk */}
                  <mesh position={[0, 0, 0]}>
                    <cylinderGeometry args={[0.3, 0.3, 2, 8]} />
                    <meshStandardMaterial color={0x4a2a1a} roughness={0.9} />
                  </mesh>
                  {/* Foliage */}
                  <mesh position={[0, 1.5, 0]}>
                    <coneGeometry args={[1.5, 3, 8]} />
                    <meshStandardMaterial color={0x2d5016} roughness={0.8} />
                  </mesh>
                </group>
              </mesh>
            </RigidBody>
          );
        });
      }, [heightData])}

      {/* Platforms/Steps for vertical gameplay */}
      <RigidBody type="fixed" colliders="cuboid">
        <mesh position={[-5, 3, 0]} castShadow receiveShadow>
          <boxGeometry args={[4, 0.5, 3]} />
          <meshStandardMaterial color={0x5a5a5a} roughness={0.8} />
        </mesh>
      </RigidBody>

      <RigidBody type="fixed" colliders="cuboid">
        <mesh position={[5, 5, 0]} castShadow receiveShadow>
          <boxGeometry args={[4, 0.5, 3]} />
          <meshStandardMaterial color={0x5a5a5a} roughness={0.8} />
        </mesh>
      </RigidBody>

      <RigidBody type="fixed" colliders="cuboid">
        <mesh position={[-10, 7, 0]} castShadow receiveShadow>
          <boxGeometry args={[3, 0.5, 3]} />
          <meshStandardMaterial color={0x5a5a5a} roughness={0.8} />
        </mesh>
      </RigidBody>

      {/* Dynamic physics objects - Pushable cubes (small, character can push) */}
      {useMemo(() => {
        // Function to sample height from heightmap
        const getHeightAt = (worldX, worldZ) => {
          const size = 500;
          const segments = 200;
          const vertexCount = segments + 1;

          const nx = (worldX + size / 2) / size;
          const nz = (worldZ + size / 2) / size;

          const clampedX = Math.max(0, Math.min(1, nx));
          const clampedZ = Math.max(0, Math.min(1, nz));

          const xIndex = Math.floor(clampedX * segments);
          const zIndex = Math.floor(clampedZ * segments);

          const x0 = Math.max(0, Math.min(segments, xIndex));
          const x1 = Math.max(0, Math.min(segments, xIndex + 1));
          const z0 = Math.max(0, Math.min(segments, zIndex));
          const z1 = Math.max(0, Math.min(segments, zIndex + 1));

          const h00 = heightData[z0 * vertexCount + x0];
          const h10 = heightData[z0 * vertexCount + x1];
          const h01 = heightData[z1 * vertexCount + x0];
          const h11 = heightData[z1 * vertexCount + x1];

          const fx = (clampedX * segments) - xIndex;
          const fz = (clampedZ * segments) - zIndex;

          const h0 = h00 * (1 - fx) + h10 * fx;
          const h1 = h01 * (1 - fx) + h11 * fx;
          const height = h0 * (1 - fz) + h1 * fz;

          return height;
        };

        const cubePositions = [-3, 2, 8, 12];
        return cubePositions.map((x, i) => {
          const terrainHeight = getHeightAt(x, 0);
          return (
            <RigidBody
              key={`cube-${i}`}
              colliders="cuboid"
              position={[x, terrainHeight + 0.15, 0]}
              friction={0.8}
              restitution={0.2}
              mass={0.5}
            >
              <mesh castShadow receiveShadow>
                <boxGeometry args={[0.3, 0.3, 0.3]} />
                <meshStandardMaterial color={0xff6b6b} roughness={0.5} />
              </mesh>
            </RigidBody>
          );
        });
      }, [heightData])}

      {/* Dynamic physics objects - Pushable spheres (light and bouncy) */}
      {useMemo(() => {
        const getHeightAt = (worldX, worldZ) => {
          const size = 500;
          const segments = 200;
          const vertexCount = segments + 1;

          const nx = (worldX + size / 2) / size;
          const nz = (worldZ + size / 2) / size;

          const clampedX = Math.max(0, Math.min(1, nx));
          const clampedZ = Math.max(0, Math.min(1, nz));

          const xIndex = Math.floor(clampedX * segments);
          const zIndex = Math.floor(clampedZ * segments);

          const x0 = Math.max(0, Math.min(segments, xIndex));
          const x1 = Math.max(0, Math.min(segments, xIndex + 1));
          const z0 = Math.max(0, Math.min(segments, zIndex));
          const z1 = Math.max(0, Math.min(segments, zIndex + 1));

          const h00 = heightData[z0 * vertexCount + x0];
          const h10 = heightData[z0 * vertexCount + x1];
          const h01 = heightData[z1 * vertexCount + x0];
          const h11 = heightData[z1 * vertexCount + x1];

          const fx = (clampedX * segments) - xIndex;
          const fz = (clampedZ * segments) - zIndex;

          const h0 = h00 * (1 - fx) + h10 * fx;
          const h1 = h01 * (1 - fx) + h11 * fx;
          const height = h0 * (1 - fz) + h1 * fz;

          return height;
        };

        const spherePositions = [25, 28, 32];
        return spherePositions.map((x, i) => {
          const terrainHeight = getHeightAt(x, 0);
          return (
            <RigidBody
              key={`sphere-${i}`}
              colliders="ball"
              position={[x, terrainHeight + 0.2, 0]}
              friction={0.3}
              restitution={0.6}
              mass={0.3}
            >
              <mesh castShadow receiveShadow>
                <sphereGeometry args={[0.2, 32, 32]} />
                <meshStandardMaterial
                  color={0x4ecdc4}
                  roughness={0.3}
                  metalness={0.5}
                />
              </mesh>
            </RigidBody>
          );
        });
      }, [heightData])}

      {/* Medium pushable crate (heavier but still pushable) */}
      {useMemo(() => {
        const getHeightAt = (worldX, worldZ) => {
          const size = 500;
          const segments = 200;
          const vertexCount = segments + 1;

          const nx = (worldX + size / 2) / size;
          const nz = (worldZ + size / 2) / size;

          const clampedX = Math.max(0, Math.min(1, nx));
          const clampedZ = Math.max(0, Math.min(1, nz));

          const xIndex = Math.floor(clampedX * segments);
          const zIndex = Math.floor(clampedZ * segments);

          const x0 = Math.max(0, Math.min(segments, xIndex));
          const x1 = Math.max(0, Math.min(segments, xIndex + 1));
          const z0 = Math.max(0, Math.min(segments, zIndex));
          const z1 = Math.max(0, Math.min(segments, zIndex + 1));

          const h00 = heightData[z0 * vertexCount + x0];
          const h10 = heightData[z0 * vertexCount + x1];
          const h01 = heightData[z1 * vertexCount + x0];
          const h11 = heightData[z1 * vertexCount + x1];

          const fx = (clampedX * segments) - xIndex;
          const fz = (clampedZ * segments) - zIndex;

          const h0 = h00 * (1 - fx) + h10 * fx;
          const h1 = h01 * (1 - fx) + h11 * fx;
          const height = h0 * (1 - fz) + h1 * fz;

          return height;
        };

        const terrainHeight = getHeightAt(45, 0);
        return (
          <RigidBody
            colliders="cuboid"
            position={[45, terrainHeight + 0.3, 0]}
            friction={0.9}
            restitution={0.1}
            mass={1.5}
          >
            <mesh castShadow receiveShadow>
              <boxGeometry args={[0.6, 0.6, 0.6]} />
              <meshStandardMaterial color={0x8b5a00} roughness={0.9} />
            </mesh>
          </RigidBody>
        );
      }, [heightData])}

      {/* Grass component */}
      {(() => {
        const enabled = grassControls?.enabled ?? false;
        if (!enabled) return null;

        const cgq3 = grassControls || {};

        return (
          <ClaudeGrassQuick3
            key="grass"
            playerPosition={playerPosRef.current}
            terrainSize={cgq3.terrainSize ?? 100}
            heightScale={cgq3.heightScale ?? 1}
            heightOffset={cgq3.heightOffset ?? 0}
            grassWidth={cgq3.grassWidth ?? 0.02}
            grassHeight={cgq3.grassHeight ?? 0.25}
            lodDistance={cgq3.lodDistance ?? 15}
            maxDistance={cgq3.maxDistance ?? 100}
            patchSize={cgq3.patchSize ?? 10}
            gridSize={cgq3.gridSize ?? 16}
            patchSpacing={cgq3.patchSpacing ?? 10}
            windEnabled={cgq3.windEnabled ?? true}
            windStrength={cgq3.windStrength ?? 1.25}
            windDirectionScale={cgq3.windDirectionScale ?? 0.05}
            windDirectionSpeed={cgq3.windDirectionSpeed ?? 0.05}
            windStrengthScale={cgq3.windStrengthScale ?? 0.25}
            windStrengthSpeed={cgq3.windStrengthSpeed ?? 1.0}
            playerInteractionEnabled={cgq3.playerInteractionEnabled ?? true}
            playerInteractionRepel={cgq3.playerInteractionRepel ?? true}
            playerInteractionRange={cgq3.playerInteractionRange ?? 2.5}
            playerInteractionStrength={cgq3.playerInteractionStrength ?? 0.2}
            playerInteractionHeightThreshold={
              cgq3.playerInteractionHeightThreshold ?? 3.0
            }
            baseColor1={cgq3.baseColor1 ?? "#051303"}
            baseColor2={cgq3.baseColor2 ?? "#061a03"}
            tipColor1={cgq3.tipColor1 ?? "#a6cc40"}
            tipColor2={cgq3.tipColor2 ?? "#cce666"}
            gradientCurve={cgq3.gradientCurve ?? 4.0}
            aoEnabled={cgq3.aoEnabled ?? true}
            aoIntensity={cgq3.aoIntensity ?? 1.0}
            grassMiddleBrightnessMin={cgq3.grassMiddleBrightnessMin ?? 0.85}
            grassMiddleBrightnessMax={cgq3.grassMiddleBrightnessMax ?? 1.0}
            fogEnabled={cgq3.fogEnabled ?? false}
            fogNear={cgq3.fogNear ?? 5.0}
            fogFar={cgq3.fogFar ?? 50.0}
            fogIntensity={cgq3.fogIntensity ?? 1.0}
            fogColor={cgq3.fogColor ?? "#4f74af"}
            specularEnabled={cgq3.specularEnabled ?? false}
            specularIntensity={cgq3.specularIntensity ?? 2.0}
            specularColor={cgq3.specularColor ?? "#ffffff"}
            specularDirectionX={cgq3.specularDirectionX ?? -1.0}
            specularDirectionY={cgq3.specularDirectionY ?? 1.0}
            specularDirectionZ={cgq3.specularDirectionZ ?? 0.5}
            backscatterEnabled={cgq3.backscatterEnabled ?? true}
            backscatterIntensity={cgq3.backscatterIntensity ?? 0.5}
            backscatterColor={cgq3.backscatterColor ?? "#51cc66"}
            backscatterPower={cgq3.backscatterPower ?? 2.0}
            frontScatterStrength={cgq3.frontScatterStrength ?? 0.3}
            rimSSSStrength={cgq3.rimSSSStrength ?? 0.5}
            grassDensity={cgq3.grassDensity ?? 3072}
          />
        );
      })()}

      {/* GrassClaude7 component */}
      {(grassClaude7Controls?.grassClaude7?.grassClaude7Enabled ||
        grassClaude7Controls?.grassClaude7Enabled) && (
        <GrassField
          gridSize={15}
          patchSpacing={8}
          centerPosition={[0, 0, 0]}
          playerPosition={playerPosRef.current}
          renderDistance={80}
          patchSize={8}
          grassWidth={0.03}
          grassHeight={0.5}
          heightmap={heightmapTexture}
          terrainHeight={15}
          terrainOffset={0}
          terrainSize={500}
          fogEnabled={false}
        />
      )}
    </group>
  );
};

