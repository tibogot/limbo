import { useRef, useMemo, useEffect } from "react";
import * as THREE from "three";

const GRASS_BLADES = 1024;
const GRASS_BLADE_VERTICES = 15;

// Helper function to generate random number in range
function randRange(min, max) {
  return Math.random() * (max - min) + min;
}

export default function Grass({
  heightData,
  terrainSize = 500,
  terrainSegments = 200,
  terrainOffset = -150,
  terrainHeight = 15,
}) {
  const meshRef = useRef();

  const { geometry, matrices } = useMemo(() => {
    // Create the actual blade shape - 15 vertices forming a triangular blade
    // This creates a blade that's narrow at bottom, widens slightly, then tapers to a point
    const bladeWidth = 0.02;
    const bladeHeight = 0.4;

    const vertices = new Float32Array([
      // Bottom vertices (base of blade) - 3 vertices
      -bladeWidth,
      0,
      0, // 0: left base
      0,
      0,
      0, // 1: center base
      bladeWidth,
      0,
      0, // 2: right base

      // Lower section - 3 vertices
      -bladeWidth * 0.9,
      bladeHeight * 0.2,
      0, // 3
      0,
      bladeHeight * 0.2,
      0, // 4
      bladeWidth * 0.9,
      bladeHeight * 0.2,
      0, // 5

      // Middle section - 3 vertices
      -bladeWidth * 0.7,
      bladeHeight * 0.5,
      0, // 6
      0,
      bladeHeight * 0.5,
      0, // 7
      bladeWidth * 0.7,
      bladeHeight * 0.5,
      0, // 8

      // Upper section - 3 vertices
      -bladeWidth * 0.4,
      bladeHeight * 0.8,
      0, // 9
      0,
      bladeHeight * 0.8,
      0, // 10
      bladeWidth * 0.4,
      bladeHeight * 0.8,
      0, // 11

      // Tip - 3 vertices (converging to point)
      -bladeWidth * 0.1,
      bladeHeight * 0.95,
      0, // 12
      0,
      bladeHeight,
      0, // 13: tip
      bladeWidth * 0.1,
      bladeHeight * 0.95,
      0, // 14
    ]);

    // Create indices to form triangles from these vertices
    const indices = new Uint16Array([
      // Bottom section triangles
      0, 1, 3, 1, 4, 3, 1, 2, 4, 2, 5, 4,

      // Lower-middle section
      3, 4, 6, 4, 7, 6, 4, 5, 7, 5, 8, 7,

      // Middle-upper section
      6, 7, 9, 7, 10, 9, 7, 8, 10, 8, 11, 10,

      // Upper-tip section
      9, 10, 12, 10, 13, 12, 10, 11, 13, 11, 14, 13,
    ]);

    // Create base geometry
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(vertices, 3));
    geo.setIndex(new THREE.BufferAttribute(indices, 1));

    // Compute bounding box and sphere for proper culling
    geo.computeBoundingBox();
    geo.computeBoundingSphere();

    // Function to sample height from heightData at given world coordinates
    // Fastest method: direct array lookup with bilinear interpolation
    const getHeightAt = (worldX, worldZ) => {
      if (!heightData) return 0; // Fallback if no height data

      const size = terrainSize;
      const segments = terrainSegments;
      const vertexCount = segments + 1;
      const terrainShift = terrainOffset;

      // Convert world coordinates to normalized coordinates (0-1)
      // Account for terrain mesh position shift
      const nx = (worldX + size / 2) / size; // -250 to 250 -> 0 to 1
      const nz = (worldZ - terrainShift + size / 2) / size; // Adjusted for mesh position

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

    // Create instance matrices for positioning grass blades
    const NUM_GRASS_X = 32;
    const NUM_GRASS_Y = 32;
    const GRASS_PATCH_SIZE = 10;

    const matrices = [];
    const matrix = new THREE.Matrix4();

    for (let i = 0; i < NUM_GRASS_X; ++i) {
      const x = (i / NUM_GRASS_X) * GRASS_PATCH_SIZE - GRASS_PATCH_SIZE * 0.5;
      for (let j = 0; j < NUM_GRASS_Y; ++j) {
        const z = (j / NUM_GRASS_Y) * GRASS_PATCH_SIZE - GRASS_PATCH_SIZE * 0.5;

        // Random position offset
        const posX = x + randRange(-0.2, 0.2);
        const posZ = z + randRange(-0.2, 0.2);

        // Sample height from terrain - FAST: one-time CPU calculation during setup
        const posY = getHeightAt(posX, posZ);

        // Optional: Add random rotation around Y axis for variety
        const rotationY = randRange(0, Math.PI * 2);

        // Set position and rotation in matrix
        // Y is the vertical axis in Three.js, so grass stands on XZ plane
        matrix.makeRotationY(rotationY);
        matrix.setPosition(posX, posY, posZ);

        matrices.push(matrix.clone());
      }
    }

    return { geometry: geo, matrices };
  }, [heightData, terrainSize, terrainSegments, terrainOffset]);

  // Set instance matrices on the mesh ref
  useEffect(() => {
    if (meshRef.current && matrices) {
      const actualCount = Math.min(matrices.length, GRASS_BLADES);

      matrices.forEach((matrix, i) => {
        if (i < GRASS_BLADES) {
          meshRef.current.setMatrixAt(i, matrix);
        }
      });

      // Update instance count to actual number of matrices
      meshRef.current.count = actualCount;
      meshRef.current.instanceMatrix.needsUpdate = true;

      // Calculate proper bounding box for all instances to prevent incorrect culling
      const expandedBox = new THREE.Box3();
      const bladeBounds = geometry.boundingBox;
      const bladeSize = bladeBounds
        ? bladeBounds.getSize(new THREE.Vector3())
        : new THREE.Vector3(0.02, 0.4, 0.02);
      const halfSize = bladeSize.clone().multiplyScalar(0.5);

      matrices.forEach((matrix) => {
        const position = new THREE.Vector3();
        const scale = new THREE.Vector3();
        const quaternion = new THREE.Quaternion();
        matrix.decompose(position, quaternion, scale);

        // Add corners of blade bounding box at this position
        const corners = [
          new THREE.Vector3(-halfSize.x, 0, -halfSize.z),
          new THREE.Vector3(halfSize.x, 0, -halfSize.z),
          new THREE.Vector3(-halfSize.x, bladeSize.y, -halfSize.z),
          new THREE.Vector3(halfSize.x, bladeSize.y, -halfSize.z),
          new THREE.Vector3(-halfSize.x, 0, halfSize.z),
          new THREE.Vector3(halfSize.x, 0, halfSize.z),
          new THREE.Vector3(-halfSize.x, bladeSize.y, halfSize.z),
          new THREE.Vector3(halfSize.x, bladeSize.y, halfSize.z),
        ];

        corners.forEach((corner) => {
          corner.applyQuaternion(quaternion).add(position);
          expandedBox.expandByPoint(corner);
        });
      });

      // Update geometry bounding box and sphere
      meshRef.current.geometry.boundingBox = expandedBox;
      const center = expandedBox.getCenter(new THREE.Vector3());
      const size = expandedBox.getSize(new THREE.Vector3());
      const radius = Math.max(size.x, size.y, size.z) * 0.5;
      meshRef.current.geometry.boundingSphere = new THREE.Sphere(
        center,
        radius
      );
    }
  }, [matrices, geometry]);

  return (
    <instancedMesh
      ref={meshRef}
      args={[geometry, null, GRASS_BLADES]}
      frustumCulled={false}
    >
      <meshBasicMaterial
        color="green"
        side={THREE.DoubleSide}
        depthWrite={true}
        depthTest={true}
      />
    </instancedMesh>
  );
}
