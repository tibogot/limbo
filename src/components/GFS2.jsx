import { useRef, useMemo } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";

const GRASS_BLADES = 4096; // 64x64 = 4,096 blades (cranked up from 32x32 = 1,024)
const GRASS_SEGMENTS = 6; // Number of segments in the blade
const GRASS_VERTICES = (GRASS_SEGMENTS + 1) * 2; // Vertices per blade (front and back)
const GRASS_PATCH_SIZE = 10;

// Helper function to generate random number in range
function randRange(min, max) {
  return Math.random() * (max - min) + min;
}

const vertexShader = `
varying vec3 vColor;
varying float vHeightPercent;
varying vec3 vRotatedNormal1;
varying vec3 vRotatedNormal2;
varying float vWidthPercent;
varying vec3 vViewPosition;
varying vec3 vUpVectorViewSpace; // Up vector transformed to view space

attribute float vertIndex;
attribute vec3 instancePosition; // Instanced attribute for grass blade position (uint16, half float encoded)
// Note: Can't use 'position' - it's reserved by Three.js, so we use 'instancePosition'

uniform float time;
uniform vec2 grassSize;

// Decode half float (16-bit) to float32
// Input is a float that was converted from uint16 (half float encoded)
// When WebGL reads uint16 as float, it converts the uint value directly (e.g., 15360 -> 15360.0)
// So we need to treat the float value as if it were the original uint16 value
float unpackHalf1x16(float v) {
  // Treat the float value as the original uint16 value
  // Since float can represent uint16 values exactly (up to 2^24), this works
  uint bits = uint(v); // Cast float to uint (extracts the uint16 value)
  uint sign = (bits >> 15u) & 1u;
  uint exponent = (bits >> 10u) & 31u;
  uint mantissa = bits & 1023u;
  
  // Handle zero
  if (exponent == 0u && mantissa == 0u) {
    return sign == 1u ? -0.0 : 0.0;
  }
  
  // Handle infinity
  if (exponent == 31u && mantissa == 0u) {
    return sign == 1u ? -1.0 / 0.0 : 1.0 / 0.0;
  }
  
  // Handle NaN
  if (exponent == 31u && mantissa != 0u) {
    return 0.0 / 0.0;
  }
  
  // Normal case: convert half float to full float
  // Half: bias 15, Full: bias 127, difference: 112
  uint biasedExp = (exponent == 0u) ? 0u : (exponent + 112u);
  uint fullBits = (sign << 31u) | (biasedExp << 23u) | (mantissa << 13u);
  
  return uintBitsToFloat(fullBits);
}

// Decode half float vec3
vec3 unpackHalf3x16(vec3 v) {
  return vec3(
    unpackHalf1x16(v.x),
    unpackHalf1x16(v.y),
    unpackHalf1x16(v.z)
  );
}

// Hash function for randomness - matches tutorial (hash12 returns vec2)
vec2 hash12(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * vec3(.1031, .1030, .0973));
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.xx + p3.yz) * p3.zy);
}

// Rotation matrix around X axis - matches tutorial
mat3 rotateX(float theta) {
  float c = cos(theta);
  float s = sin(theta);
  return mat3(
    1.0, 0.0, 0.0,
    0.0, c, -s,
    0.0, s, c
  );
}

// Rotation matrix around Y axis - for normal blending
mat3 rotateY(float theta) {
  float c = cos(theta);
  float s = sin(theta);
  return mat3(
    c, 0.0, s,
    0.0, 1.0, 0.0,
    -s, 0.0, c
  );
}

// Ease out function for smooth thickening curve
float easeOut(float t, float power) {
  return pow(t, power);
}

// Ease in function for color gradient
float easeIn(float t, float power) {
  return pow(t, power);
}

void main() {
  // Decode half float position values (matches tutorial)
  // The instancePosition is encoded as uint16 (half floats), but WebGL reads it as float
  // (converting uint16 to float directly), so we need to decode it properly
  vec3 grassBladeWorldPos = unpackHalf3x16(instancePosition);
  
  // Get random values for this blade based on position (matches tutorial exactly)
  float perBladeHash = hash12(grassBladeWorldPos.xz).x;
  float randomAngle = perBladeHash * 2.0 * 3.14159; // 2 * PI radians (full rotation)
  
  // Get additional random values using hash with offsets
  vec2 hash1 = hash12(grassBladeWorldPos.xz + vec2(1.0, 0.0));
  vec2 hash2 = hash12(grassBladeWorldPos.xz + vec2(2.0, 0.0));
  float randomHeight = mix(0.75, 1.5, hash1.x);
  float randomLean = mix(0.3, 0.7, hash2.x); // Increased from 0.1-0.4 to 0.3-0.7 for more visible curve
  
  float GRASS_SEGMENTS_F = ${GRASS_SEGMENTS}.0;
  float GRASS_VERTICES_F = ${GRASS_VERTICES}.0;
  
  // Figure out which vertex this is
  float vertID = mod(vertIndex, GRASS_VERTICES_F);
  
  // Front or back side (-1 = back, 1 = front)
  float zSide = -(floor(vertIndex / GRASS_VERTICES_F) * 2.0 - 1.0);
  
  // Left or right edge (0 = left, 1 = right)
  float xSide = mod(vertID, 2.0);
  
  // Height percentage along blade (0 = base, 1 = tip)
  float heightPercent = (vertID - xSide) / (GRASS_SEGMENTS_F * 2.0);
  
  // Calculate blade dimensions
  float grassHeight = grassSize.y * randomHeight;
  float grassWidth = grassSize.x * (1.0 - heightPercent); // Taper to tip
  
  // Position of this vertex
  float x = (xSide - 0.5) * grassWidth;
  float y = heightPercent * grassHeight;
  
  // Width percent (normalized position across blade width) - for normal blending
  float widthPercent = xSide; // 0 = left edge, 1 = right edge
  
  // Calculate grass vertex normal (before transformations)
  // For a vertical blade, the normal is perpendicular to the blade surface
  // Before rotation, the blade is in the XY plane, so normal points in Z
  vec3 grassVertexNormal = vec3(0.0, 0.0, 1.0);
  
  // VERTEX SHADER: Create two rotated normals for blending (matches tutorial)
  // The rotated normals are just generated in the vertex shader by rotating 
  // slightly on the y axis (before any other transformations).
  float PI = 3.14159;
  vec3 rotatedNormal1 = rotateY(PI * 0.3) * grassVertexNormal;
  vec3 rotatedNormal2 = rotateY(PI * -0.3) * grassVertexNormal;
  
  // Apply curve/lean to blade - matches tutorial exactly!
  // Surprisingly, this works pretty ok
  float curveAmount = randomLean * heightPercent;
  
  // Create a 3x3 rotation matrix around x
  mat3 grassMat = rotateX(curveAmount);
  
  // Now generate the grass vertex position
  vec3 grassVertexPosition = grassMat * vec3(x, y, 0.0);
  
  // Apply Y rotation for blade orientation
  mat3 rotY = mat3(
    cos(randomAngle), 0.0, sin(randomAngle),
    0.0, 1.0, 0.0,
    -sin(randomAngle), 0.0, cos(randomAngle)
  );
  grassVertexPosition = rotY * grassVertexPosition;
  grassVertexPosition += grassBladeWorldPos;
  
  // IMPORTANT: Normals are NOT transformed by other rotations!
  // They are rotated ONLY by rotateY(PI * 0.3) and rotateY(PI * -0.3)
  // "before any other transformations" - matches tutorial exactly
  
  // Color gradient from base to tip - matches tutorial exactly
  // I just picked 2 colours, darker green and a yellowish colour.
  // Adjusted to more natural hues - base is rich dark green, tip is vibrant lime-yellow
  vec3 baseColour = vec3(0.08, 0.35, 0.02); // Rich dark green base
  vec3 tipColour = vec3(0.7, 0.85, 0.25); // Natural lime-yellow-green tip (more green in yellow)
  
  // Do a gradient from base to tip, controlled by shaping function.
  vec3 diffuseColour = mix(baseColour, tipColour, easeIn(heightPercent, 4.0));
  
  vColor = diffuseColour;
  vHeightPercent = heightPercent;
  
  // Transform normals to view space for lighting (normalMatrix handles non-uniform scaling)
  // The normalMatrix is the inverse transpose of the modelViewMatrix
  vec3 transformedNormal1 = normalize(normalMatrix * rotatedNormal1);
  vec3 transformedNormal2 = normalize(normalMatrix * rotatedNormal2);
  
  // Pass transformed normals and width percent to fragment shader
  vRotatedNormal1 = transformedNormal1;
  vRotatedNormal2 = transformedNormal2;
  vWidthPercent = widthPercent;
  
  // Final position in model space
  vec4 modelPosition = modelMatrix * vec4(grassVertexPosition, 1.0);
  vec4 mvPosition = viewMatrix * modelPosition;
  
  // Pass view position for distance calculation in fragment shader
  vViewPosition = mvPosition.xyz;
  
  // Transform up vector to view space for normal blending with terrain
  vec3 upVectorWorld = vec3(0.0, 1.0, 0.0); // World up vector
  vUpVectorViewSpace = normalize((viewMatrix * vec4(upVectorWorld, 0.0)).xyz); // Transform to view space
  
  // View-space thickening based on camera angle (matches tutorial exactly)
  // Calculate grass face normal (perpendicular to blade surface) in world space
  vec3 grassFaceNormal = rotY * vec3(0.0, 0.0, 1.0); // Face normal after blade orientation rotation
  
  // Transform face normal to view space
  vec3 grassFaceNormalVS = normalize((viewMatrix * vec4(grassFaceNormal, 0.0)).xyz);
  
  // Calculate view direction in view space (from vertex to camera)
  vec3 viewDir = normalize(-mvPosition.xyz);
  
  // Calculate how head-on or side-on the camera is viewing the grass blade
  // Use XZ components only (ignore vertical component)
  float viewDotNormal = dot(grassFaceNormalVS.xz, viewDir.xz);
  viewDotNormal = clamp(viewDotNormal, 0.0, 1.0); // saturate equivalent
  
  // Calculate view space thicken factor
  // When viewed head-on (viewDotNormal = 1): 1.0 - 1.0 = 0.0, no thickening
  // When viewed side-on (viewDotNormal = 0): 1.0 - 0.0 = 1.0, full thickening
  float viewSpaceThickenFactor = easeOut(1.0 - viewDotNormal, 4.0);
  
  // Refine for orthogonal views - prevent artifacts when perfectly edge-on
  viewSpaceThickenFactor *= smoothstep(0.0, 0.2, viewDotNormal);
  
  // Apply view space adjustment
  // xDirection: -0.5 for left edge, +0.5 for right edge
  float xDirection = (xSide - 0.5);
  mvPosition.x += viewSpaceThickenFactor * xDirection * grassWidth;
  
  // Final projected position
  vec4 projectedPosition = projectionMatrix * mvPosition;
  
  gl_Position = projectedPosition;
}
`;

const fragmentShader = `
varying vec3 vColor;
varying float vHeightPercent;
varying vec3 vRotatedNormal1;
varying vec3 vRotatedNormal2;
varying float vWidthPercent;
varying vec3 vViewPosition;
varying vec3 vUpVectorViewSpace; // Up vector transformed to view space

uniform bool showNormals; // Debug: show normals as color
uniform float terrainNormalBlendStart; // Distance where blending starts
uniform float terrainNormalBlendEnd; // Distance where blending ends
uniform float density; // Density in range [0, 1] - 0 = no grass, 1 = full grass

// Ease in function for AO calculation
float easeIn(float t, float power) {
  return pow(t, power);
}

void main() {
  // FRAGMENT SHADER: Blend between rotated normals for 3D appearance (matches tutorial)
  // The rotated normals are just generated
  // in the vertex shader by rotating slightly on
  // the y axis (before any any other transformations).
  
  float normalMixFactor = vWidthPercent;
  vec3 normal = mix(vRotatedNormal1, vRotatedNormal2, normalMixFactor);
  normal = normalize(normal);
  
  // Blend the normal with the up vector (terrain normal) depending on the distance
  // With the specular i just blend the normal with the up vector based on distance
  float viewDistance = length(vViewPosition);
  float normalBlendFactor = smoothstep(terrainNormalBlendStart, terrainNormalBlendEnd, viewDistance);
  
  // Blend normal with up vector - far away grass uses up vector, close uses blade normal
  normal = mix(normal, vUpVectorViewSpace, normalBlendFactor);
  normal = normalize(normal);
  
  // DEBUG: Visualize normals as color (normal ranges from -1 to 1, map to 0 to 1)
  if (showNormals) {
    vec3 normalColor = normal * 0.5 + 0.5; // Map from [-1,1] to [0,1]
    gl_FragColor = vec4(normalColor, 1.0);
    return;
  }
  
  // Use the blended normal for lighting to create rounded 3D appearance
  // Normal is already in view space from vertex shader
  // Simple directional light in view space (from top, slightly forward)
  vec3 lightDir = normalize(vec3(0.2, 1.0, 0.3));
  
  // Lambertian lighting: dot product between normal and light direction
  float NdotL = max(dot(normal, lightDir), 0.0);
  
  // Lighting to show rounded 3D effect from normal blending
  // Normals are rotated BEFORE other transformations (done in vertex shader)
  float ambient = 0.85; // Higher ambient to brighten colors (was 0.6)
  float diffuse = NdotL * 0.8; // Diffuse for rounded effect
  float ambientLighting = ambient + diffuse; // Range: 0.85 to 1.65 (brighter overall)
  
  // Ambient occlusion based on density and height
  // Density is in the range [0, 1]
  // 0 being no grass
  // 1 being full grass
  float aoForDensity = mix(1.0, 0.25, density); // Fully dense areas get more shading (lower value)
  
  // Adjust based on height - base is darker, tip is brighter
  float ao = mix(aoForDensity, 1.0, easeIn(vHeightPercent, 2.0));
  
  // Mix in the ambient occlusion term.
  ambientLighting *= ao;
  
  // Apply lighting to gradient color - the blended normals create the rounded appearance
  vec3 color = vColor * ambientLighting;
  
  gl_FragColor = vec4(color, 1.0);
}
`;

export default function Grass({
  heightData,
  terrainSize = 500,
  terrainSegments = 200,
  terrainOffset = -150,
  terrainHeight = 15,
  showNormals = false,
}) {
  const materialRef = useRef();

  const geometry = useMemo(() => {
    // Function to sample height from heightData
    const getHeightAt = (worldX, worldZ) => {
      if (!heightData) return 0;
      const size = terrainSize;
      const segments = terrainSegments;
      const vertexCount = segments + 1;
      const terrainShift = terrainOffset;
      const nx = (worldX + size / 2) / size;
      const nz = (worldZ - terrainShift + size / 2) / size;
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
      return h0 * (1 - fz) + h1 * fz;
    };
    // Create vertex ID array - each vertex gets an ID
    const vertID = new Uint8Array(GRASS_VERTICES * 2); // *2 for front and back
    for (let i = 0; i < GRASS_VERTICES * 2; ++i) {
      vertID[i] = i;
    }

    // Create indices for triangles
    const indices = [];
    for (let i = 0; i < GRASS_SEGMENTS; ++i) {
      const vi = i * 2;
      // Front face triangles
      indices.push(vi + 0, vi + 1, vi + 2);
      indices.push(vi + 2, vi + 1, vi + 3);

      // Back face triangles (reversed winding)
      const fi = GRASS_VERTICES + vi;
      indices.push(fi + 2, fi + 1, fi + 0);
      indices.push(fi + 3, fi + 1, fi + 2);
    }

    // Create offset positions for each grass blade
    const offsets = [];
    const NUM_GRASS_X = 64;
    const NUM_GRASS_Y = 64;

    for (let i = 0; i < NUM_GRASS_X; ++i) {
      const x = (i / NUM_GRASS_X) * GRASS_PATCH_SIZE - GRASS_PATCH_SIZE * 0.5;
      for (let j = 0; j < NUM_GRASS_Y; ++j) {
        const z = (j / NUM_GRASS_Y) * GRASS_PATCH_SIZE - GRASS_PATCH_SIZE * 0.5;
        const posX = x + randRange(-0.2, 0.2);
        const posZ = z + randRange(-0.2, 0.2);
        const posY = getHeightAt(posX, posZ); // Sample terrain height
        offsets.push(posX, posY, posZ);
      }
    }

    // Use Float16 (half precision) for better performance - matches tutorial exactly!
    // THREE.DataUtils.toHalfFloat() exists - use it directly (matches tutorial!)
    const offsetsData = offsets.map(THREE.DataUtils.toHalfFloat);

    // Create instanced attribute with Float16 data
    // Note: Tutorial uses InstancedFloat16BufferAttribute, but Three.js uses
    // InstancedBufferAttribute with Uint16Array (from toHalfFloat)
    const instancedOffsetAttribute = new THREE.InstancedBufferAttribute(
      new Uint16Array(offsetsData),
      3
    );

    // Create geometry
    const geo = new THREE.InstancedBufferGeometry();

    geo.setAttribute("vertIndex", new THREE.Uint8BufferAttribute(vertID, 1));
    // Set instancePosition attribute using Float16 data - matches tutorial pattern
    geo.setAttribute("instancePosition", instancedOffsetAttribute);
    geo.setIndex(new THREE.BufferAttribute(new Uint16Array(indices), 1));

    return geo;
  }, [heightData, terrainSize, terrainSegments, terrainOffset]);

  useFrame((state) => {
    if (materialRef.current) {
      materialRef.current.uniforms.time.value = state.clock.elapsedTime;
      materialRef.current.uniforms.showNormals.value = showNormals;
    }
  });

  return (
    <instancedMesh args={[geometry, null, GRASS_BLADES]} frustumCulled={false}>
      <shaderMaterial
        ref={materialRef}
        vertexShader={vertexShader}
        fragmentShader={fragmentShader}
        uniforms={{
          time: { value: 0 },
          grassSize: { value: new THREE.Vector2(0.08, 0.4) }, // Width: 0.08 (was 0.02 - too thin), Height: 0.4
          showNormals: { value: showNormals }, // Debug: toggle to visualize normals
          terrainNormalBlendStart: { value: 10.0 }, // Distance where normal blending starts
          terrainNormalBlendEnd: { value: 50.0 }, // Distance where normal blending ends
          density: { value: 1.0 }, // Density in range [0, 1] - 0 = no grass, 1 = full grass
        }}
        side={THREE.DoubleSide}
      />
    </instancedMesh>
  );
}
