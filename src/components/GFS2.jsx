import { useRef, useMemo } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { acceleratedRaycast } from "three-mesh-bvh";

const GRASS_BLADES = 4096; // Good density (64x64 = 4096)
const GRASS_BLADE_VERTICES = 15; // Matches tutorial exactly
const GRASS_PATCH_SIZE = 5; // Reduced patch size for higher density

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
attribute vec3 instancePosition; // Instanced attribute for grass blade position (can't use 'position' - it's reserved in Three.js)

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

// 2D noise function (noise12: 1D output from 2D input) - for grass animation
// Simplified version using hash directly with smooth interpolation
float noise12(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  
  // Smoothstep for smooth interpolation
  vec2 u = f * f * (3.0 - 2.0 * f);
  
  // Get hash values for the four corners
  float a = hash12(i).x;
  float b = hash12(i + vec2(1.0, 0.0)).x;
  float c = hash12(i + vec2(0.0, 1.0)).x;
  float d = hash12(i + vec2(1.0, 1.0)).x;
  
  // Bilinear interpolation
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
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

// Remap function - maps value from one range to another
float remap(float value, float inMin, float inMax, float outMin, float outMax) {
  return outMin + (value - inMin) * (outMax - outMin) / (inMax - inMin);
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
  
  float GRASS_BLADE_VERTICES_F = ${GRASS_BLADE_VERTICES}.0;
  
  // Figure out which vertex this is (matches tutorial)
  // Assuming 7 segments with 2 vertices each = 14 vertices, structure may vary
  // TODO: Need to see tutorial's vertex structure to match exactly
  float vertID = vertIndex;
  
  // Temporary structure - will update when we see tutorial's approach
  // Assuming: 7 segments (0-6), each with left (even) and right (odd) vertices
  float segmentID = floor(vertID / 2.0);
  float xSide = mod(vertID, 2.0); // 0 = left, 1 = right
  float heightPercent = segmentID / 6.0; // 0 to 1 along blade height
  
  // Calculate blade dimensions
  float grassHeight = grassSize.y * randomHeight;
  float grassWidth = grassSize.x * (1.0 - heightPercent); // Taper to tip
  
  // Position of this vertex
  // Blade is created in local space: x = width, y = height, z = 0
  // The blade stands vertically, so base is at y=0, tip is at y=grassHeight
  // IMPORTANT: The base vertex (heightPercent=0) must be at y=0 to stay on ground
  float x = (xSide - 0.5) * grassWidth; // Width (left/right)
  float y = heightPercent * grassHeight; // Height (up from base, 0 to grassHeight)
  float z = 0.0; // Blade is flat (no depth)
  
  // Ensure base is exactly at y=0 (no floating point errors)
  if (heightPercent < 0.001) {
    y = 0.0;
  }
  
  // Width percent (normalized position across blade width) - for normal blending
  float widthPercent = xSide; // 0 = left edge, 1 = right edge
  
  // Calculate grass vertex normal (before transformations)
  // For a vertical blade, the normal is perpendicular to the blade surface
  // Before rotation, the blade is in the XY plane (vertical), so normal points in Z
  vec3 grassVertexNormal = vec3(0.0, 0.0, 1.0);
  
  // VERTEX SHADER: Create two rotated normals for blending (matches tutorial)
  // The rotated normals are just generated in the vertex shader by rotating 
  // slightly on the y axis (before any other transformations).
  float PI = 3.14159;
  vec3 rotatedNormal1 = rotateY(PI * 0.3) * grassVertexNormal;
  vec3 rotatedNormal2 = rotateY(PI * -0.3) * grassVertexNormal;
  
  // Wind effects - matches tutorial exactly
  // Sample noise and then remap into the range [0, 2PI].
  float windDir = noise12(grassBladeWorldPos.xz * 0.05 + 0.05 * time);
  windDir = remap(windDir, 0.0, 1.0, -1.0, 1.0);
  windDir = remap(windDir, -1.0, 1.0, 0.0, PI * 2.0);
  
  // Another noise sample for the strength of the wind.
  float windNoiseSample = noise12(grassBladeWorldPos.xz * 0.25 + time);
  
  // Try and shape it a bit with easeIn(), this is pretty arbitrary.
  windNoiseSample = remap(windNoiseSample, 0.0, 1.0, -1.0, 1.0);
  float windLeanAngle = remap(windNoiseSample, -1.0, 1.0, 0.25, 1.0);
  windLeanAngle = easeIn(windLeanAngle, 2.0) * 1.25;
  
  // Apply curve/lean to blade - matches tutorial exactly!
  // Surprisingly, this works pretty ok
  float curveAmount = randomLean * heightPercent;
  
  // Sample noise using time + world position.
  float noiseSample = noise12(vec2(time * 0.35) + grassBladeWorldPos.xz);
  
  // Add the animated noise onto the grass curve.
  curveAmount += noiseSample * 0.1;
  
  // Apply wind rotation - creates smooth bend from base, not hook
  // Reduce wind strength and apply more subtly to avoid hook shape
  float windBend = windLeanAngle * heightPercent * 0.15; // Reduced multiplier for smoother effect
  
  // Rotate around Y axis to orient in wind direction, then X axis for the bend
  // This creates a smooth lean from base, not a hook
  mat3 windRotY = rotateY(windDir);
  mat3 windRotX = rotateX(windBend);
  mat3 windMat = windRotY * windRotX;
  
  // Create a 3x3 rotation matrix around x (for forward/backward lean)
  mat3 grassMat = rotateX(curveAmount);
  
  // Combine rotations: apply wind first, then curve
  // This creates a smooth bend from base with natural curve variation
  mat3 combinedMat = grassMat * windMat;
  
  // Now generate the grass vertex position in local space
  // Blade is created: x = width, y = height (0 to grassHeight), z = 0
  // The base is at (0, 0, 0), tip is at (0, grassHeight, 0)
  vec3 localPosition = vec3(x, y, z);
  
  // Apply rotations to local position (rotations happen around origin/base)
  vec3 rotatedPosition = combinedMat * localPosition;
  
  // Apply Y rotation for blade orientation (after wind and curve)
  mat3 rotY = mat3(
    cos(randomAngle), 0.0, sin(randomAngle),
    0.0, 1.0, 0.0,
    -sin(randomAngle), 0.0, cos(randomAngle)
  );
  rotatedPosition = rotY * rotatedPosition;
  
  // Now add world position - this places the blade in world space
  // The base (which should be at (0,0,0) after rotations) will be at grassBladeWorldPos
  vec3 grassVertexPosition = rotatedPosition + grassBladeWorldPos;
  
  // CRITICAL: Ensure base vertex Y is exactly at terrain height
  // After rotations, the base Y might have moved slightly, so we force it to the terrain height
  // But we keep X and Z as they are (with blade width) to maintain the blade shape at the base
  if (heightPercent < 0.001) {
    // Base vertex: only force Y to terrain height, keep X and Z for blade width
    grassVertexPosition.y = grassBladeWorldPos.y;
  }
  
  // IMPORTANT: Normals are NOT transformed by other rotations!
  // They are rotated ONLY by rotateY(PI * 0.3) and rotateY(PI * -0.3)
  // "before any other transformations" - matches tutorial exactly
  
  // Color gradient from base to tip - matches tutorial exactly
  // I just picked 2 colours, darker green and a yellowish colour.
  // Lighter green color matching #59b01c (RGB: 89, 176, 28)
  vec3 baseColour = vec3(0.349, 0.690, 0.110); // Lighter green base color #59b01c
  vec3 tipColour = vec3(0.733, 0.831, 0.475); // Tip color #bbd479 (RGB: 187, 212, 121)
  
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
  
  // Final position - apply modelMatrix then viewMatrix
  // The grassVertexPosition includes the world position, but we still need modelMatrix
  // for the mesh's own transform (position, rotation, scale)
  vec4 worldPos = vec4(grassVertexPosition, 1.0);
  vec4 mvPosition = viewMatrix * modelMatrix * worldPos;
  
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
  
  // DEBUG: Test if time is updating - uncomment to see time as color (should pulse)
  // Uncomment the next 2 lines to test if time is updating:
  // float timeFrac = fract(time * 0.1);
  // gl_FragColor = vec4(timeFrac, timeFrac, timeFrac, 1.0); return;
  
  // Use the blended normal for lighting to create rounded 3D appearance
  // Normal is already in view space from vertex shader
  // Simple directional light in view space (from top, slightly forward)
  vec3 lightDir = normalize(vec3(0.2, 1.0, 0.3));
  
  // Lambertian lighting: dot product between normal and light direction
  float NdotL = max(dot(normal, lightDir), 0.0);
  
  // Lighting to show rounded 3D effect from normal blending
  // Normals are rotated BEFORE other transformations (done in vertex shader)
  float ambient = 1.0; // Very high ambient to match tutorial's bright appearance (was 0.85)
  float diffuse = NdotL * 1.0; // Full diffuse for rounded effect (was 0.8)
  float ambientLighting = ambient + diffuse; // Range: 1.0 to 2.0 (very bright overall)
  
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
  terrainMesh = null, // Terrain mesh for BVH raycasting
}) {
  const materialRef = useRef();

  const geometry = useMemo(() => {
    // Function to get height using BVH raycasting (preferred method)
    const getHeightAtBVH = (worldX, worldZ) => {
      if (!terrainMesh) return 0;

      // Enable accelerated raycasting if not already enabled
      if (!THREE.Mesh.prototype.raycast.isAcceleratedRaycast) {
        THREE.Mesh.prototype.raycast = acceleratedRaycast;
      }

      const raycaster = new THREE.Raycaster();
      const rayOrigin = new THREE.Vector3(worldX, 100, worldZ); // Start high above
      const rayDirection = new THREE.Vector3(0, -1, 0); // Cast downward

      raycaster.set(rayOrigin, rayDirection);
      raycaster.firstHitOnly = true;

      const intersects = raycaster.intersectObject(terrainMesh, false);

      if (intersects.length > 0) {
        return intersects[0].point.y;
      }

      // Fallback to heightData if BVH fails
      return getHeightAtHeightData(worldX, worldZ);
    };

    // Fallback function to sample height from heightData
    const getHeightAtHeightData = (worldX, worldZ) => {
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

    // Use BVH if terrainMesh is available, otherwise fallback to heightData
    const getHeightAt = terrainMesh ? getHeightAtBVH : getHeightAtHeightData;
    // Create offset positions for each grass blade - matches tutorial exactly
    const offsets = [];
    const NUM_GRASS_X = 64; // 64x64 = 4096 blades (good density)
    const NUM_GRASS_Y = 64;

    for (let i = 0; i < NUM_GRASS_X; ++i) {
      const x = i / NUM_GRASS_Y - 0.5; // Matches tutorial: (i / NUM_GRASS_Y) - 0.5
      for (let j = 0; j < NUM_GRASS_X; ++j) {
        const z = j / NUM_GRASS_Y - 0.5; // Matches tutorial: (j / NUM_GRASS_Y) - 0.5 (this is Z in world space)
        const worldX = x * GRASS_PATCH_SIZE + randRange(-0.2, 0.2);
        const worldZ = z * GRASS_PATCH_SIZE + randRange(-0.2, 0.2);
        const worldY = getHeightAt(worldX, worldZ); // Calculate height using BVH or heightData
        offsets.push(worldX); // X position
        offsets.push(worldY); // Y position (calculated from terrain)
        offsets.push(worldZ); // Z position
      }
    }

    // Convert to half float - matches tutorial exactly
    const offsetsData = offsets.map(THREE.DataUtils.toHalfFloat);

    // Create vertID array - matches tutorial exactly
    const vertID = new Uint8Array(GRASS_BLADE_VERTICES);
    for (let i = 0; i < GRASS_BLADE_VERTICES; ++i) {
      vertID[i] = i;
    }

    // Create index buffer - temporary until we see tutorial's CreateIndexBuffer()
    // Assuming 7 segments (0-6) with 2 vertices each = 14 vertices, plus 1 tip = 15
    // Create triangles for each segment
    const indices = [];
    for (let i = 0; i < 6; ++i) {
      const vi = i * 2; // Base vertex index for this segment
      // Front face triangles
      indices.push(vi + 0, vi + 1, vi + 2);
      indices.push(vi + 2, vi + 1, vi + 3);
    }
    // Add tip triangle if needed
    if (GRASS_BLADE_VERTICES >= 14) {
      indices.push(12, 13, 14); // Tip triangle
    }

    // Create geometry - matches tutorial exactly
    const geo = new THREE.InstancedBufferGeometry();
    geo.instanceCount = GRASS_BLADES; // Matches tutorial
    geo.setAttribute("vertIndex", new THREE.Uint8BufferAttribute(vertID, 1));
    // Note: Tutorial uses InstancedFloat16BufferAttribute, but Three.js r180 uses InstancedBufferAttribute
    // Note: Can't use 'position' as attribute name - it's reserved in Three.js
    // Using 'instancePosition' instead, but functionality matches tutorial
    geo.setAttribute(
      "instancePosition",
      new THREE.InstancedBufferAttribute(new Uint16Array(offsetsData), 3)
    );
    // Set index buffer - temporary until we see tutorial's CreateIndexBuffer()
    geo.setIndex(new THREE.BufferAttribute(new Uint16Array(indices), 1));

    return geo;
  }, [heightData, terrainSize, terrainSegments, terrainOffset, terrainMesh]);

  // Memoize uniforms - only create once, update values via ref in useFrame
  const uniforms = useMemo(
    () => ({
      time: { value: 0 }, // Initial value, will be updated via ref
      grassSize: { value: new THREE.Vector2(0.04, 0.25) }, // Width: 0.04 (reduced from 0.05), Height: 0.25
      showNormals: { value: showNormals },
      terrainNormalBlendStart: { value: 10.0 },
      terrainNormalBlendEnd: { value: 50.0 },
      density: { value: 1.0 },
    }),
    [showNormals] // Only recreate if showNormals changes
  );

  useFrame((state) => {
    // CRITICAL: Update uniforms directly via ref every frame
    // This is the only way to ensure time updates in the shader
    if (materialRef.current && materialRef.current.uniforms) {
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
        uniforms={uniforms}
        side={THREE.DoubleSide}
      />
    </instancedMesh>
  );
}
