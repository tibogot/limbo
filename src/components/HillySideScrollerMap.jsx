import { RigidBody } from "@react-three/rapier";
import { useFrame } from "@react-three/fiber";
import { useMemo, useRef, useState, useEffect } from "react";
import { createNoise2D } from "simplex-noise";
import * as THREE from "three";
import {
  computeBoundsTree,
  disposeBoundsTree,
  acceleratedRaycast,
} from "three-mesh-bvh";

export const HillySideScrollerMap = ({
  playerPosition = [0, 0, 0],
  grassControls,
  onTerrainReady,
  ...props
}) => {
  const groundRef = useRef();
  const playerPosRef = useRef(new THREE.Vector3(...playerPosition));
  const [terrainReady, setTerrainReady] = useState(false);
  const terrainBVHRef = useRef(null);
  const terrainMeshRef = useRef(null);
  const getHeightAtWorldPositionRef = useRef(null);

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
  const { terrainGeometry, terrainMaterial, heightData, heightmapTexture } =
    useMemo(() => {
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

        // Map to world coordinates for 2.5D side-scroller
        // X: -250 to 250 (left-right, centered)
        // Z: -250 to 250 (will be shifted by mesh position)
        // Terrain mesh will be positioned to place playable area at front
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

        // For 2.5D side view: hills should be flatter near character (world Z=0)
        // and more prominent in background (negative Z, further from camera)
        // Character is at world Z=0, which is local Z=150 (mesh is at Z=-150)
        // Terrain mesh is positioned at Z=-150, so world Z=0 = local Z=150

        const meshZOffset = 150; // Mesh is positioned at Z=-150, so add 150 to get world Z
        const worldZPosition = worldZ + meshZOffset; // Convert local Z to world Z

        // Keep it very flat near character (world Z=0 area, which is local Z=150)
        const flatZone = 15; // Keep flat within 15 units of world Z=0
        if (Math.abs(worldZPosition) < flatZone) {
          // Very flat near character - fade from 0 to small height
          const flatFactor = Math.abs(worldZPosition) / flatZone; // 0 at center, 1 at edge
          height *= flatFactor * 0.15; // Very flat near character
        } else if (worldZPosition < 0) {
          // Background (negative world Z) - hills increase as we go further back
          const backDistance = Math.abs(worldZPosition) - flatZone; // Distance beyond flat zone
          const maxBackDistance = size / 2 - flatZone; // Maximum distance to edge
          const normalizedBack = Math.min(backDistance / maxBackDistance, 1);
          // Hills increase more quickly toward background (lower exponent = faster increase)
          const fadeFactor = Math.pow(normalizedBack, 0.5);
          height *= 0.15 + fadeFactor * 0.85; // Start from flat zone height, increase to full
        } else {
          // Foreground (positive world Z) - keep relatively flat
          const frontDistance = worldZPosition - flatZone;
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
      const textureData = new Float32Array(
        heightmapResolution * heightmapResolution
      );

      // Fill texture data by sampling heightData with bilinear interpolation
      // Normalize height to 0-1 range based on maxHeight (not min/max of actual data)
      // This ensures the shader can correctly scale it back using terrainHeight
      // Note: Terrain is shifted backward by 150 units, so Z ranges from -400 to 100
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

          textureData[y * heightmapResolution + x] = Math.max(
            0,
            Math.min(1, height)
          );
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

      return {
        terrainGeometry: geometry,
        terrainMaterial: material,
        heightData,
        heightmapTexture,
      };
    }, []);

  // Build BVH for terrain mesh
  useEffect(() => {
    if (terrainGeometry) {
      // Enable accelerated raycasting
      THREE.BufferGeometry.prototype.computeBoundsTree = computeBoundsTree;
      THREE.BufferGeometry.prototype.disposeBoundsTree = disposeBoundsTree;
      THREE.Mesh.prototype.raycast = acceleratedRaycast;

      // Build BVH for the terrain geometry
      if (!terrainGeometry.boundsTree) {
        terrainGeometry.computeBoundsTree();
      }
      terrainBVHRef.current = terrainGeometry.boundsTree;

      // Wait a bit for mesh to be ready
      const timer = setTimeout(() => {
        setTerrainReady(true);
        // Expose height function to parent
        if (onTerrainReady && getHeightAtWorldPositionRef.current) {
          onTerrainReady(getHeightAtWorldPositionRef.current);
        }
      }, 100);
      return () => clearTimeout(timer);
    }
  }, [terrainGeometry, onTerrainReady]);

  // Function to get height at world position using BVH raycast
  const getHeightAtWorldPosition = useMemo(() => {
    const fn = (worldX, worldY, worldZ) => {
      if (!terrainBVHRef.current || !terrainMeshRef.current) {
        return worldY; // Fallback to input Y if BVH not ready
      }

      const raycaster = new THREE.Raycaster();
      const rayOrigin = new THREE.Vector3(worldX, worldY + 100, worldZ); // Start high above
      const rayDirection = new THREE.Vector3(0, -1, 0); // Cast downward

      raycaster.set(rayOrigin, rayDirection);
      raycaster.firstHitOnly = true;

      const intersects = raycaster.intersectObject(
        terrainMeshRef.current,
        false
      );

      if (intersects.length > 0) {
        return intersects[0].point.y;
      }

      return worldY; // Fallback if no intersection
    };
    getHeightAtWorldPositionRef.current = fn;
    return fn;
  }, []);

  return (
    <group {...props}>
      {/* 500x500 Procedural terrain with hills */}
      {/* Hills are flatter near character (Z=0) and more prominent in background */}
      <RigidBody type="fixed" colliders="trimesh">
        <mesh
          ref={(mesh) => {
            groundRef.current = mesh;
            if (mesh) {
              terrainMeshRef.current = mesh;
            }
          }}
          geometry={terrainGeometry}
          material={terrainMaterial}
          rotation={[-Math.PI / 2, 0, 0]}
          position={[0, 0, -150]}
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
          const terrainShift = -150; // Terrain is shifted backward by 150 units

          // Convert world coordinates to normalized coordinates (0-1)
          // Account for terrain mesh position shift: mesh is at Z=-150
          const nx = (worldX + size / 2) / size; // -250 to 250 -> 0 to 1
          const nz = (worldZ - terrainShift + size / 2) / size; // Adjusted for mesh position shift

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
          const fx = clampedX * segments - xIndex;
          const fz = clampedZ * segments - zIndex;

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
          const terrainShift = -150; // Terrain is shifted backward by 150 units

          const nx = (worldX + size / 2) / size;
          const nz = (worldZ - terrainShift + size / 2) / size; // Adjusted for terrain shift

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

          const fx = clampedX * segments - xIndex;
          const fz = clampedZ * segments - zIndex;

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

          const fx = clampedX * segments - xIndex;
          const fz = clampedZ * segments - zIndex;

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

          const fx = clampedX * segments - xIndex;
          const fz = clampedZ * segments - zIndex;

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
    </group>
  );
};
