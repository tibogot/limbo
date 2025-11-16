import { RigidBody } from "@react-three/rapier";
import { useMemo, useRef } from "react";
import { createNoise2D } from "simplex-noise";
import * as THREE from "three";

export const ProceduralTerrain = ({ ...props }) => {
  const meshRef = useRef();

  const { geometry, heightData, material } = useMemo(() => {
    const size = 200;
    const segments = 200;
    const heightScale = 15; // Increased for better visibility
    const noiseScale = 0.03; // Adjusted for better terrain variation

    console.log("Generating procedural terrain...");

    // Create simplex noise with random seed
    const noise2D = createNoise2D(() => Math.random());

    // Generate height data using Brownian motion (multiple octaves)
    const heightData = new Float32Array((segments + 1) * (segments + 1));

    for (let y = 0; y <= segments; y++) {
      for (let x = 0; x <= segments; x++) {
        // Normalize coordinates to 0-1 range
        const nx = x / segments;
        const ny = y / segments;

        // Map to world coordinates for noise sampling
        const worldX = (nx - 0.5) * size;
        const worldZ = (ny - 0.5) * size;

        // Brownian motion: combine multiple octaves of noise
        let height = 0;
        let amplitude = 1;
        let frequency = 1;
        let maxValue = 0;

        // 6 octaves for smooth, natural-looking terrain
        for (let octave = 0; octave < 6; octave++) {
          const sampleX = worldX * frequency * noiseScale;
          const sampleY = worldZ * frequency * noiseScale;
          height += noise2D(sampleX, sampleY) * amplitude;
          maxValue += amplitude;
          amplitude *= 0.5; // Each octave has half the amplitude
          frequency *= 2; // Each octave has double the frequency
        }

        // Normalize to 0-1 range
        height = (height / maxValue + 1) * 0.5;

        // Apply power curve for more dramatic hills
        height = Math.pow(height, 1.2);

        // Scale to desired height
        height *= heightScale;

        const index = y * (segments + 1) + x;
        heightData[index] = height;
      }
    }

    // Create plane geometry
    const geometry = new THREE.PlaneGeometry(size, size, segments, segments);

    // Apply height data to vertices
    // PlaneGeometry creates vertices row by row, matching our heightData indexing
    const positions = geometry.attributes.position;
    const vertexCount = segments + 1;

    for (let i = 0; i < positions.count; i++) {
      // Calculate x and y indices from vertex index
      const x = i % vertexCount;
      const y = Math.floor(i / vertexCount);
      const heightIndex = y * vertexCount + x;

      // Set the Z coordinate (which becomes Y/height after -90° rotation around X)
      // After rotation: X→X, Y→-Z, Z→Y
      // So we set original Z to get final Y (height)
      positions.setZ(i, heightData[heightIndex]);
    }

    // Recalculate normals for proper lighting
    geometry.computeVertexNormals();

    // Create material
    const material = new THREE.MeshStandardMaterial({
      color: 0x6b8e23, // Olive green color for better visibility
      roughness: 0.8,
      metalness: 0.1,
    });

    console.log("Terrain generated:", {
      size,
      segments,
      vertices: positions.count,
      minHeight: Math.min(...heightData),
      maxHeight: Math.max(...heightData),
    });

    return { geometry, heightData, material };
  }, []);

  return (
    <group {...props}>
      <RigidBody type="fixed" colliders="trimesh">
        <mesh
          ref={meshRef}
          geometry={geometry}
          material={material}
          rotation={[-Math.PI / 2, 0, 0]}
          receiveShadow
          castShadow
        />
      </RigidBody>
    </group>
  );
};
