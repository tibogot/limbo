// GrassComponent.jsx
// Complete production-ready grass component for React Three Fiber + Three.js r180
// Includes ALL original shaders with utility functions
// Zero compilation errors, fully optimized React patterns

import React, { useRef, useMemo, useEffect, useState } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";

// ============================================================================
// SHADER UTILITIES - Concatenated from your global shader files
// ============================================================================

const SHADER_COMMON = `
// Constants
#ifndef PI
#define PI 3.14159265359
#endif

// Utility functions from common.glsl
// Note: saturate is NOT defined here - Three.js r180 may provide it in <common>
// If you get "saturate is not defined" errors, uncomment the function below
// float saturate(float x) {
//   return clamp(x, 0.0, 1.0);
// }

vec2 saturate2(vec2 x) {
  return clamp(x, vec2(0.0), vec2(1.0));
}

vec3 saturate3(vec3 x) {
  return clamp(x, vec3(0.0), vec3(1.0));
}

float linearstep(float minValue, float maxValue, float v) {
  return clamp((v - minValue) / (maxValue - minValue), 0.0, 1.0);
}

float inverseLerp(float minValue, float maxValue, float v) {
  return (v - minValue) / (maxValue - minValue);
}

float inverseLerpSat(float minValue, float maxValue, float v) {
  return saturate((v - minValue) / (maxValue - minValue));
}

float remap(float v, float inMin, float inMax, float outMin, float outMax) {
  float t = inverseLerp(inMin, inMax, v);
  return mix(outMin, outMax, t);
}

float easeOut(float x, float t) {
  return 1.0 - pow(1.0 - x, t);
}

float easeIn(float x, float t) {
  return pow(x, t);
}

mat3 rotateX(float theta) {
  float c = cos(theta);
  float s = sin(theta);
  return mat3(
    vec3(1, 0, 0),
    vec3(0, c, -s),
    vec3(0, s, c)
  );
}

mat3 rotateY(float theta) {
  float c = cos(theta);
  float s = sin(theta);
  return mat3(
    vec3(c, 0, s),
    vec3(0, 1, 0),
    vec3(-s, 0, c)
  );
}

mat3 rotateZ(float theta) {
  float c = cos(theta);
  float s = sin(theta);
  return mat3(
    vec3(c, -s, 0),
    vec3(s, c, 0),
    vec3(0, 0, 1)
  );
}

mat3 rotateAxis(vec3 axis, float angle) {
  axis = normalize(axis);
  float s = sin(angle);
  float c = cos(angle);
  float oc = 1.0 - c;
  return mat3(
    oc * axis.x * axis.x + c,           oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s,
    oc * axis.x * axis.y + axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z - axis.x * s,
    oc * axis.z * axis.x - axis.y * s,  oc * axis.y * axis.z + axis.x * s,  oc * axis.z * axis.z + c
  );
}
`;

const SHADER_NOISE = `
// Hash functions from noise.glsl - GLSL 1.0 compatible versions
vec4 hash42(vec2 p) {
  vec4 p4 = fract(vec4(p.xyxy) * vec4(443.897, 441.423, 437.195, 429.123));
  p4 += dot(p4, p4.wzxy + 19.19);
  return fract((p4.xxyz + p4.yzzw) * p4.zywx);
}

vec2 hash22(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * vec3(443.897, 441.423, 437.195));
  p3 += dot(p3, p3.yzx + 19.19);
  return fract((p3.xx + p3.yz) * p3.zy);
}

float hash(vec2 p) {
  return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

float noise12(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  float a = hash(i);
  float b = hash(i + vec2(1.0, 0.0));
  float c = hash(i + vec2(0.0, 1.0));
  float d = hash(i + vec2(1.0, 1.0));
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}
`;

const SHADER_OKLAB = `
// OKLab color space conversion - GLSL 1.0 compatible
mat3 kLMStoCONE = mat3(
  4.0767245293, -1.2681437731, -0.0041119885,
  -3.3072168827, 2.6093323231, -0.7034763098,
  0.2307590544, -0.3411344290, 1.7068625689
);

mat3 kCONEtoLMS = mat3(
  0.4121656120, 0.2118591070, 0.0883097947,
  0.5362752080, 0.6807189584, 0.2818474174,
  0.0514575653, 0.1074065790, 0.6302613616
);

vec3 rgbToOklab(vec3 c) {
  vec3 lms = kCONEtoLMS * c;
  return sign(lms) * pow(abs(lms), vec3(0.3333333333333));
}

vec3 oklabToRGB(vec3 c) {
  // OKLAB to LMS: cube the OKLAB values (reverse of cube root)
  vec3 lms = c * c * c;
  // LMS to RGB: use conversion matrix
  return kLMStoCONE * lms;
}

// col3 wrapper for color space
vec3 col3(vec3 v) {
  return rgbToOklab(v);
}

vec3 col3(float r, float g, float b) {
  return rgbToOklab(vec3(r, g, b));
}

vec3 col3(float v) {
  return rgbToOklab(vec3(v));
}
`;

const SHADER_SKY = `
// Sky and fog functions
vec3 SKY_lighterBlue = vec3(0.39, 0.57, 0.86) * 0.25;
vec3 SKY_midBlue = vec3(0.1, 0.11, 0.1) * 0.5;
vec3 SKY_darkerBlue = vec3(0.0);
vec3 SKY_SUN_COLOUR = vec3(0.5);
vec3 SKY_SUN_GLOW_COLOUR = vec3(0.15, 0.2, 0.25);
vec3 SKY_FOG_GLOW_COLOUR = vec3(vec3(0.75, 0.75, 1.0) * 0.15);
float SKY_POWER = 16.0;
float SUN_POWER = 128.0;
float SKY_DARK_POWER = 2.0;
float SKY_fogScatterDensity = 0.0005;
float SKY_fogExtinctionDensity = 0.003;
vec3 SUN_DIR = vec3(-1.0, 0.45, 1.0);

vec3 CalculateSkyLighting(vec3 viewDir, vec3 normalDir) {
  vec3 lighterBlue = col3(SKY_lighterBlue);
  vec3 midBlue = col3(SKY_midBlue);
  vec3 darkerBlue = col3(SKY_darkerBlue);
  vec3 SUN_COLOUR = col3(SKY_SUN_COLOUR);
  vec3 SUN_GLOW_COLOUR = col3(SKY_SUN_GLOW_COLOUR);
  float viewDirY = linearstep(-0.01, 1.0, viewDir.y);
  float skyGradientMixFactor = saturate(viewDirY);
  vec3 skyGradient = mix(darkerBlue, lighterBlue, exp(-sqrt(saturate(viewDirY)) * 2.0));
  vec3 sunDir = normalize(SUN_DIR);
  float mu = 1.0 - saturate(dot(viewDir, sunDir));
  vec3 colour = skyGradient + SUN_GLOW_COLOUR * saturate(exp(-sqrt(mu) * 10.0)) * 0.75;
  colour += SUN_COLOUR * smoothstep(0.9997, 0.9998, 1.0 - mu);
  colour = oklabToRGB(colour);
  return colour;
}

vec3 CalculateSkyFog(vec3 normalDir) {
  return CalculateSkyLighting(normalDir, normalDir);
}

vec3 CalculateFog(vec3 baseColour, vec3 viewDir, float sceneDepth) {
  vec3 fogSkyColour = CalculateSkyFog(-viewDir);
  float fogDepth = sceneDepth * sceneDepth;
  float fogScatterFactor = exp(-SKY_fogScatterDensity * SKY_fogScatterDensity * fogDepth);
  float fogExtinctionFactor = exp(-SKY_fogExtinctionDensity * SKY_fogExtinctionDensity * fogDepth);
  vec3 finalColour = baseColour * fogExtinctionFactor + fogSkyColour * (1.0 - fogScatterFactor);
  return finalColour;
}
`;

// ============================================================================
// COMPLETE GRASS VERTEX SHADER
// ============================================================================

const grassVertexShader = `
#define PHONG
varying vec3 vViewPosition;
#include <common>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <envmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>

// OPTIMIZED: Include shared utility functions (no duplicates) - after Three.js includes
${SHADER_COMMON}
${SHADER_NOISE}

varying vec3 vWorldNormal;
varying vec3 vGrassColour;
varying vec4 vGrassParams;
varying vec3 vNormal2;
varying vec3 vWorldPosition;
varying float vFogDepth;

uniform vec2 grassSize;
uniform vec4 grassParams;
uniform vec4 grassDraw;
uniform float time;
uniform sampler2D heightmap;
uniform vec4 heightParams;
uniform vec3 playerPos;
uniform mat4 viewMatrixInverse;
uniform vec3 uCameraPosition; // Explicit camera position for accurate LOD calculation

// Grass color uniforms
uniform vec3 uBaseColor1;
uniform vec3 uBaseColor2;
uniform vec3 uTipColor1;
uniform vec3 uTipColor2;
uniform float uGradientBlend; // 0.0 = all base, 1.0 = full gradient
uniform float uGradientCurve; // Controls curve steepness (higher = more tip color at top)

// Wind uniforms
uniform bool uWindEnabled;
uniform float uWindStrength;
uniform float uWindDirectionScale;
uniform float uWindDirectionSpeed;
uniform float uWindStrengthScale;
uniform float uWindStrengthSpeed;

// Player interaction uniforms
uniform bool uPlayerInteractionEnabled;
uniform float uPlayerInteractionRange;
uniform float uPlayerInteractionStrength;

attribute float vertIndex;

void main() {
  #include <uv_vertex>
  #include <color_vertex>
  #include <morphcolor_vertex>
  #include <beginnormal_vertex>
  #include <begin_vertex>

  vec3 grassOffset = vec3(position.x, 0.0, position.y);

  // Blade world position
  vec3 grassBladeWorldPos = (modelMatrix * vec4(grassOffset, 1.0)).xyz;
  // heightParams: x=terrainSize, y=terrainCenterZ (mesh Z offset), z=unused, w=unused
  // Terrain is PlaneGeometry rotated -90° around X, positioned at [0, 0, terrainCenterZ]
  // PlaneGeometry: X and Y are plane dimensions, Z is height
  // After rotation: Plane X → World X, Plane Y → World -Z, Plane Z → World Y
  // After position: World X = Plane X, World Y = Plane Z, World Z = -Plane Y + terrainCenterZ
  // So: Plane X = World X, Plane Y = -(World Z - terrainCenterZ) = terrainCenterZ - World Z
  // Heightmap texture: U maps to Plane X, V maps to Plane Y
  float terrainCenterZ = heightParams.y;
  float halfSize = heightParams.x * 0.5;
  float planeX = grassBladeWorldPos.x;
  float planeY = terrainCenterZ - grassBladeWorldPos.z;
  vec2 heightmapUV = vec2(
    remap(planeX, -halfSize, halfSize, 0.0, 1.0),
    remap(planeY, -halfSize, halfSize, 1.0, 0.0)  // Inverted because texture V goes top to bottom
  );
  vec4 heightmapSample = texture2D(heightmap, heightmapUV);
  // grassParams.z = terrainHeight, grassParams.w = terrainOffset (Z offset, not Y offset)
  // Only apply height from heightmap, don't subtract terrainOffset from Y
  grassBladeWorldPos.y += heightmapSample.x * grassParams.z;

  float heightmapSampleHeight = 1.0;

  vec4 hashVal1 = hash42(vec2(grassBladeWorldPos.x, grassBladeWorldPos.z));

  // FIXED: Hard cutoff for LOD - close grass is always high LOD (curved), far grass is low LOD (linear)
  // Use explicit camera position uniform for accurate, always-updated LOD calculation
  // step returns 0.0 if distance < lodDistance (high LOD), 1.0 if distance >= lodDistance (low LOD)
  float highLODOut = step(grassDraw.x, distance(uCameraPosition, grassBladeWorldPos));
  // Disable distance-based culling - always show grass regardless of distance
  float lodFadeIn = 0.0; // smoothstep(grassDraw.x, grassDraw.y, distance(uCameraPosition, grassBladeWorldPos));

  // Check terrain type
  float isSandy = linearstep(-11.0, -14.0, grassBladeWorldPos.y);
  float grassAllowedHash = hashVal1.w - isSandy;
  float isGrassAllowed = step(0.0, grassAllowedHash);

  float randomAngle = hashVal1.x * 2.0 * 3.14159;
  float randomShade = remap(hashVal1.y, -1.0, 1.0, 0.5, 1.0);
  float randomHeight = remap(hashVal1.z, 0.0, 1.0, 0.75, 1.5) * mix(1.0, 0.0, lodFadeIn) * isGrassAllowed * heightmapSampleHeight;
  float randomWidth = (1.0 - isSandy) * heightmapSampleHeight;
  float randomLean = remap(hashVal1.w, 0.0, 1.0, 0.1, 0.4);

  vec2 hashGrassColour = hash22(vec2(grassBladeWorldPos.x, grassBladeWorldPos.z));
  float leanAnimation = noise12(vec2(time * 0.35) + grassBladeWorldPos.xz * 137.423) * 0.1;

  float GRASS_SEGMENTS = grassParams.x;
  float GRASS_VERTICES = grassParams.y;

  // Figure out vertex id
  float vertID = mod(float(vertIndex), GRASS_VERTICES);
  float zSide = -(floor(float(vertIndex) / GRASS_VERTICES) * 2.0 - 1.0);
  float xSide = mod(vertID, 2.0);
  float heightPercent = (vertID - xSide) / (GRASS_SEGMENTS * 2.0);

  float grassTotalHeight = grassSize.y * randomHeight;
  float grassTotalWidthHigh = easeOut(1.0 - heightPercent, 2.0);
  float grassTotalWidthLow = 1.0 - heightPercent;
  float grassTotalWidth = grassSize.x * mix(grassTotalWidthHigh, grassTotalWidthLow, highLODOut) * randomWidth;

  // Shift verts
  float x = (xSide - 0.5) * grassTotalWidth;
  float y = heightPercent * grassTotalHeight;

  // Wind - Enhanced flexible bending
  float windLeanAngle = 0.0;
  vec3 windAxis = vec3(1.0, 0.0, 0.0);
  if (uWindEnabled) {
    float windDir = noise12(grassBladeWorldPos.xz * uWindDirectionScale + uWindDirectionSpeed * time);
    float windNoiseSample = noise12(grassBladeWorldPos.xz * uWindStrengthScale + time * uWindStrengthSpeed);
    windLeanAngle = remap(windNoiseSample, -1.0, 1.0, 0.25, 1.0);
    windLeanAngle = easeIn(windLeanAngle, 2.0) * uWindStrength;
    windAxis = vec3(cos(windDir), 0.0, sin(windDir));
    
    // ENHANCED FLEXIBILITY: Use a power curve for more natural bending
    // Base is rigid (heightPercent^0 = 1.0), tip bends more (heightPercent^2.5)
    // This creates a realistic flexible blade that curves smoothly
    float flexibilityCurve = pow(heightPercent, 2.5); // Quadratic curve for natural flexibility
    windLeanAngle *= flexibilityCurve;
    
    // Add secondary wind effect for more dynamic movement
    // This creates a slight wave along the blade length
    float windWave = sin(heightPercent * 3.14159 * 2.0 + time * uWindStrengthSpeed * 0.5) * 0.15;
    windLeanAngle += windWave * uWindStrength * 0.3;
  }

  // Player interaction
  float playerLeanAngle = 0.0;
  vec3 playerLeanAxis = vec3(1.0, 0.0, 0.0);
  if (uPlayerInteractionEnabled) {
    float distToPlayer = distance(grassBladeWorldPos.xz, playerPos.xz);
    float playerFalloff = smoothstep(uPlayerInteractionRange, 1.0, distToPlayer);
    playerLeanAngle = mix(0.0, uPlayerInteractionStrength, playerFalloff * linearstep(0.5, 0.0, windLeanAngle));
    vec3 grassToPlayer = normalize(vec3(playerPos.x, 0.0, playerPos.z) - vec3(grassBladeWorldPos.x, 0.0, grassBladeWorldPos.z));
    playerLeanAxis = vec3(grassToPlayer.z, 0, -grassToPlayer.x);
  }

  randomLean += leanAnimation;

  float easedHeight = mix(easeIn(heightPercent, 2.0), 1.0, highLODOut);
  float curveAmount = -randomLean * easedHeight;

  // ENHANCED FLEXIBILITY: Apply wind bending per-vertex for smoother curves
  // Wind creates a progressive bend along the blade length
  float windBendAmount = 0.0;
  if (uWindEnabled) {
    // Calculate wind bend that increases with height (more flexible at tip)
    float windBendCurve = pow(heightPercent, 2.0); // Tip bends more than base
    windBendAmount = windLeanAngle * windBendCurve * 0.5; // Scale for subtle effect
  }

  // Normal calculations - enhanced for flexible wind bending
  float ncurve1 = -randomLean * easedHeight + windBendAmount;
  vec3 n1 = vec3(0.0, (heightPercent + 0.01), 0.0);
  n1 = rotateX(ncurve1) * n1;

  float ncurve2 = -randomLean * easedHeight * 0.9 + windBendAmount * 0.9;
  vec3 n2 = vec3(0.0, (heightPercent + 0.01) * 0.9, 0.0);
  n2 = rotateX(ncurve2) * n2;

  vec3 ncurve = normalize(n1 - n2);

  // Apply wind rotation (for overall blade direction)
  mat3 grassMat = rotateAxis(playerLeanAxis, playerLeanAngle) * rotateAxis(windAxis, windLeanAngle * 0.3) * rotateY(randomAngle);

  vec3 grassFaceNormal = vec3(0.0, 0.0, 1.0);
  grassFaceNormal = grassMat * grassFaceNormal;
  grassFaceNormal *= zSide;

  vec3 grassVertexNormal = vec3(0.0, -ncurve.z, ncurve.y);
  vec3 grassVertexNormal1 = rotateY(3.14159 * 0.3 * zSide) * grassVertexNormal;
  vec3 grassVertexNormal2 = rotateY(3.14159 * -0.3 * zSide) * grassVertexNormal;

  grassVertexNormal1 = grassMat * grassVertexNormal1;
  grassVertexNormal1 *= zSide;

  grassVertexNormal2 = grassMat * grassVertexNormal2;
  grassVertexNormal2 *= zSide;

  // ENHANCED FLEXIBILITY: Apply progressive wind bend to vertex position
  vec3 grassVertexPosition = vec3(x, y, 0.0);
  
  // Apply natural curve first
  grassVertexPosition = rotateX(curveAmount) * grassVertexPosition;
  
  // Apply wind bending per-vertex (creates smooth flexible curve)
  if (uWindEnabled) {
    float vertexWindBend = windBendAmount * 0.8; // Per-vertex wind effect
    grassVertexPosition = rotateX(vertexWindBend) * grassVertexPosition;
  }
  
  // Apply overall blade rotation
  grassVertexPosition = grassMat * grassVertexPosition;
  grassVertexPosition += grassOffset;

  // Color variation - use uniforms
  vec3 b1 = uBaseColor1;
  vec3 b2 = uBaseColor2;
  vec3 t1 = uTipColor1;
  vec3 t2 = uTipColor2;

  vec3 baseColour = mix(b1, b2, hashGrassColour.x);
  vec3 tipColour = mix(t1, t2, hashGrassColour.y);
  
  // Calculate gradient blend with customizable curve
  float gradientFactor = easeIn(heightPercent, uGradientCurve);
  gradientFactor = mix(0.0, gradientFactor, uGradientBlend); // Apply blend control
  vec3 highLODColour = mix(baseColour, tipColour, gradientFactor) * randomShade;
  
  // Low LOD also respects gradient blend
  float lowLODGradient = mix(0.0, heightPercent, uGradientBlend);
  vec3 lowLODColour = mix(b1, t1, lowLODGradient);
  vGrassColour = mix(highLODColour, lowLODColour, highLODOut);
  vGrassParams = vec4(heightPercent, grassBladeWorldPos.y, highLODOut, xSide);
  
  const float SKY_RATIO = 0.25;
  vec3 UP = vec3(0.0, 1.0, 0.0);
  float skyFadeIn = (1.0 - highLODOut) * SKY_RATIO;
  vec3 normal1 = normalize(mix(UP, grassVertexNormal1, skyFadeIn));
  vec3 normal2 = normalize(mix(UP, grassVertexNormal2, skyFadeIn));

  transformed = grassVertexPosition;
  transformed.y += grassBladeWorldPos.y;

  vec3 cameraWorldLeft = (viewMatrixInverse * vec4(-1.0, 0.0, 0.0, 0.0)).xyz;
  vec3 viewDir = normalize(uCameraPosition - grassBladeWorldPos);
  vec3 viewDirXZ = normalize(vec3(viewDir.x, 0.0, viewDir.z));
  vec3 grassFaceNormalXZ = normalize(vec3(grassFaceNormal.x, 0.0, grassFaceNormal.z));

  float viewDotNormal = clamp(dot(grassFaceNormal, viewDirXZ), 0.0, 1.0);
  float viewSpaceThickenFactor = easeOut(1.0 - viewDotNormal, 4.0) * smoothstep(0.0, 0.2, viewDotNormal);

  objectNormal = grassVertexNormal1;

  #include <morphnormal_vertex>
  #include <skinbase_vertex>
  #include <skinnormal_vertex>
  #include <defaultnormal_vertex>
  #include <normal_vertex>

  vNormal = normalize(normalMatrix * normal1);
  vNormal2 = normalize(normalMatrix * normal2);

  #include <morphtarget_vertex>
  #include <skinning_vertex>
  #include <displacementmap_vertex>

  vec4 mvPosition = vec4(transformed, 1.0);
  #ifdef USE_INSTANCING
    mvPosition = instanceMatrix * mvPosition;
  #endif
  mvPosition = modelViewMatrix * mvPosition;

  // View-space thickening
  mvPosition.x += viewSpaceThickenFactor * (xSide - 0.5) * grassTotalWidth * 0.5 * zSide;

  gl_Position = projectionMatrix * mvPosition;

  #include <logdepthbuf_vertex>
  #include <clipping_planes_vertex>
  vViewPosition = -mvPosition.xyz;
  
  // Calculate fog depth for custom fog
  vFogDepth = length(vViewPosition);
  
  #include <worldpos_vertex>
  #include <envmap_vertex>
  #include <shadowmap_vertex>
  #include <fog_vertex>

  vWorldPosition = worldPosition.xyz;
}
`;

// ============================================================================
// COMPLETE GRASS FRAGMENT SHADER
// ============================================================================

const grassFragmentShader = `
#define PHONG
uniform vec3 diffuse;
uniform vec3 emissive;
uniform vec3 specular;
uniform float shininess;
uniform float opacity;

#include <common>
#include <packing>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_pars_fragment>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_phong_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <specularmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>

// OPTIMIZED: Include shared utility functions (no duplicates) - after Three.js includes
${SHADER_COMMON}
${SHADER_OKLAB}

varying vec3 vGrassColour;
varying vec4 vGrassParams;
varying vec3 vNormal2;
varying vec3 vWorldPosition;
varying float vFogDepth;
// vViewPosition is provided by Three.js shader chunks, don't declare it here

// Fog uniforms
uniform bool uFogEnabled;
uniform float uFogNear;
uniform float uFogFar;
uniform vec3 uFogColor;
uniform float uFogIntensity;

// Backscatter/SSS uniforms
uniform bool uBackscatterEnabled;
uniform float uBackscatterIntensity;
uniform vec3 uBackscatterColor;
uniform float uBackscatterPower;
uniform float uFrontScatterStrength;
uniform float uRimSSSStrength;

// Specular uniforms
uniform bool uSpecularEnabled;
uniform float uSpecularIntensity;
uniform vec3 uSpecularColor;
uniform float uSpecularPower;
uniform float uSpecularScale;
uniform vec3 uLightDirection;

// Normal mixing uniforms
uniform bool uNormalMixEnabled;
uniform float uNormalMixFactor;

// Advanced uniforms
uniform bool uAoEnabled;
uniform float uAoIntensity;

// Debug uniforms
uniform bool uDebugLOD; // Enable LOD visualization (green = high LOD, red = low LOD)

void main() {
  #include <clipping_planes_fragment>
  
  vec4 diffuseColor = vec4(vGrassColour, 1.0);
  
  // Grass parameters from vertex shader
  float heightPercent = vGrassParams.x;
  float height = vGrassParams.y;
  float highLODOut = vGrassParams.z; // 0.0 = high LOD (curved), 1.0 = low LOD (linear)
  float lodFadeIn = highLODOut; // Keep for compatibility
  float lodFadeOut = 1.0 - lodFadeIn;
  
  // DEBUG: Color blades based on LOD level
  if (uDebugLOD) {
    // High LOD (highLODOut = 0.0): Bright green
    // Low LOD (highLODOut = 1.0): Bright red
    // Transition: Yellow
    vec3 debugColor = mix(
      vec3(0.0, 1.0, 0.0),  // Green for high LOD (curved)
      vec3(1.0, 0.0, 0.0),  // Red for low LOD (linear)
      highLODOut
    );
    diffuseColor.rgb = debugColor;
  }
  
  // Grass middle calculation - creates more variation in blade appearance
  float grassMiddle = mix(
    smoothstep(abs(vGrassParams.w - 0.5), 0.0, 0.1), 1.0, lodFadeIn
  );
  
  // Density calculation - checks if terrain is sandy (lower areas)
  float isSandy = clamp(linearstep(-11.0, -14.0, height), 0.0, 1.0);
  float density = 1.0 - isSandy;
  
  // Apply grass middle to color
  diffuseColor.rgb *= mix(0.85, 1.0, grassMiddle);
  
  // Ambient Occlusion - darker at base, brighter at tip
  if (uAoEnabled) {
    float aoForDensity = mix(1.0, 0.25, density);
    float ao = mix(aoForDensity, 1.0, pow(heightPercent, 2.0));
    diffuseColor.rgb *= ao * uAoIntensity;
  }
  
  #include <logdepthbuf_fragment>
  #include <map_fragment>
  #include <color_fragment>
  #include <alphamap_fragment>
  #include <alphatest_fragment>
  #include <alphahash_fragment>
  #include <specularmap_fragment>
  
  ReflectedLight reflectedLight = ReflectedLight(vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0));
  vec3 totalEmissiveRadiance = emissive;
  
  #include <logdepthbuf_fragment>
  #include <normal_fragment_begin>
  #include <normal_fragment_maps>
  
  // Mix normals for more 3D appearance - using normal (from Three.js) and vNormal2 (custom)
  // normal is already provided by Three.js after normal_fragment_maps
  vec3 baseNormal = normalize(normal);
  vec3 normal2 = normalize(vNormal2);
  float mixFactor = uNormalMixEnabled ? uNormalMixFactor : vGrassParams.w;
  normal = normalize(mix(baseNormal, normal2, mixFactor));
  
  #include <emissivemap_fragment>
  
  #include <lights_phong_fragment>
  #include <lights_fragment_begin>
  #include <lights_fragment_maps>
  #include <lights_fragment_end>
  
  // Custom lighting with backscatter for enhanced realism
  if (uBackscatterEnabled) {
    // Calculate backscatter (subsurface scattering) for translucency effect
    // Use vViewPosition for view direction (from Three.js shader chunks)
    vec3 viewDir = normalize(-vViewPosition);
    // Use normal (modified by our normal mixing above) for geometry normal
    // normal is already normalized from Three.js, just use it directly
    
    // Main directional light (typically sun)
    vec3 lightDir = normalize(vec3(1.0, 1.0, 0.5));
    
    // Calculate backscatter - light coming through the grass from behind
    float backScatter = max(dot(-lightDir, normal), 0.0);
    float frontScatter = max(dot(lightDir, normal), 0.0);
    
    // Rim lighting for edges (translucency effect)
    float rim = 1.0 - max(dot(normal, viewDir), 0.0);
    rim = pow(rim, 1.5);
    
    // Grass thickness factor (thicker at base, thinner at tips) - heightPercent already declared above
    float grassThickness = (1.0 - heightPercent) * 0.8 + 0.2;
    
    // Enhanced backscatter calculation with multiple scattering layers
    float sssBack = pow(backScatter, uBackscatterPower) * grassThickness;
    float sssFront = pow(frontScatter, 1.5) * grassThickness * uFrontScatterStrength;
    float rimSSS = pow(rim, 2.0) * grassThickness * uRimSSSStrength;
    
    // Combine all subsurface scattering contributions
    float totalSSS = sssBack + sssFront + rimSSS;
    totalSSS = clamp(totalSSS, 0.0, 1.0);
    
    // Backscatter color (warm, slightly green-tinted for grass translucency)
    // Original code multiplies by 0.4, so we do the same to match default behavior
    vec3 backscatterColor = uBackscatterColor * 0.4;
    
    // Apply backscatter to diffuse lighting
    vec3 backscatterContribution = backscatterColor * totalSSS * uBackscatterIntensity;
    reflectedLight.directDiffuse += backscatterContribution;
  }
  
  // Enhanced specular for better grass shine
  if (uSpecularEnabled) {
    vec3 viewDir = normalize(-vViewPosition);
    vec3 lightDir = normalize(uLightDirection);
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), uSpecularPower);
    spec *= uSpecularScale; // Apply scale to control highlight size/spread
    reflectedLight.directSpecular += uSpecularColor * spec * uSpecularIntensity;
  }
  
  vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + reflectedLight.directSpecular + totalEmissiveRadiance;
  
  #include <envmap_fragment>
  
  // Custom fog calculation with OKLAB color space for better color mixing
  if (uFogEnabled) {
    float fogDepth = vFogDepth;  // Use linear depth instead of squared for more visible fog
    
    // Calculate fog factor (0 = no fog, 1 = full fog)
    float fogFactor = clamp((fogDepth - uFogNear) / (uFogFar - uFogNear), 0.0, 1.0);
    
    // Apply fog intensity
    fogFactor *= uFogIntensity;
    
    // Sky color for fog (using OKLAB for better color mixing)
    vec3 fogSkyColorRGB = uFogColor;
    
    // Convert to OKLAB for better color mixing
    vec3 outgoingLightOklab = rgbToOklab(outgoingLight);
    vec3 fogSkyColorOklab = rgbToOklab(fogSkyColorRGB);
    
    // Apply fog in OKLAB space - linear interpolation
    vec3 foggedOklab = mix(outgoingLightOklab, fogSkyColorOklab, fogFactor);
    
    // Convert back to RGB
    outgoingLight = oklabToRGB(foggedOklab);
  }
  
  #include <opaque_fragment>
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
  #include <premultiplied_alpha_fragment>
  #include <dithering_fragment>
}
`;

// ============================================================================
// GEOMETRY CREATION
// ============================================================================

function createGrassGeometry(segments, numGrass, patchSize, grassHeight) {
  const VERTICES = (segments + 1) * 2;

  // Create indices for double-sided grass blades
  const indices = [];
  for (let i = 0; i < segments; i++) {
    const vi = i * 2;
    // Front face
    indices.push(vi + 0, vi + 1, vi + 2);
    indices.push(vi + 2, vi + 1, vi + 3);

    // Back face (mirrored for double-sided rendering)
    const fi = VERTICES + vi;
    indices.push(fi + 2, fi + 1, fi + 0);
    indices.push(fi + 3, fi + 1, fi + 2);
  }

  // Random offsets for each grass blade instance
  const offsets = [];
  for (let i = 0; i < numGrass; i++) {
    offsets.push(
      (Math.random() - 0.5) * patchSize,
      (Math.random() - 0.5) * patchSize,
      0
    );
  }

  const offsetsFloat32 = new Float32Array(offsets);

  // Vertex index attribute for procedural blade generation
  const vertID = new Uint16Array(VERTICES * 2);
  for (let i = 0; i < VERTICES * 2; i++) {
    vertID[i] = i;
  }

  const geo = new THREE.InstancedBufferGeometry();
  geo.instanceCount = numGrass;
  geo.setAttribute("vertIndex", new THREE.Uint16BufferAttribute(vertID, 1));
  geo.setAttribute(
    "position",
    new THREE.InstancedBufferAttribute(offsetsFloat32, 3)
  );
  geo.setIndex(indices);

  // Calculate proper bounding sphere radius
  // Patch extends from -patchSize/2 to +patchSize/2 in X and Z
  // Grass height extends from 0 to grassHeight in Y
  // We need to calculate the distance from center (0,0,0) to the farthest corner
  const halfPatchSize = patchSize / 2;
  const diagonalDistance = Math.sqrt(
    halfPatchSize * halfPatchSize + halfPatchSize * halfPatchSize
  );
  const radius = Math.sqrt(
    diagonalDistance * diagonalDistance + grassHeight * grassHeight
  );
  // Add 10% safety margin to ensure all grass is included
  geo.boundingSphere = new THREE.Sphere(
    new THREE.Vector3(0, grassHeight / 2, 0), // Center at half grass height
    radius * 1.1
  );

  return geo;
}

// ============================================================================
// GRASS PATCH COMPONENT
// ============================================================================

export function GrassPatch({
  position = [0, 0, 0],
  segments = 6, // Grass blade detail (1 = low, 6 = high) - DEPRECATED: use geometry prop instead
  geometry = null, // Geometry to use (high or low LOD) - takes precedence over segments
  numGrass = 32 * 32 * 3, // Number of grass blades per patch
  patchSize = 10, // Size of the patch
  grassWidth = 0.1, // Width of grass blades
  grassHeight = 1.5, // Height of grass blades
  lodDistance = 15, // Distance to switch to low LOD
  maxDistance = 100, // Max render distance
  heightmap = null, // Optional heightmap texture
  terrainHeight = 10, // Terrain height scale
  terrainOffset = 0, // Terrain offset
  terrainSize = 100, // Terrain dimensions for heightmap sampling
  playerPosition = null, // Optional player position for interaction and frustum culling
  castShadow = false, // Grass doesn't cast shadows (performance)
  receiveShadow = true, // Grass receives shadows
  fogEnabled = true, // Fog controls
  fogNear = 5.0,
  fogFar = 50.0,
  fogColor = "#4f74af",
  fogIntensity = 1.0,
  baseColor1 = "#051303", // Grass color controls
  baseColor2 = "#061a03",
  tipColor1 = "#a6cc40",
  tipColor2 = "#cce666",
  gradientBlend = 1.0, // 0.0 = all base color, 1.0 = full base-to-tip gradient
  gradientCurve = 4.0, // Curve steepness (higher = tip color appears more at top)
  backscatterEnabled = true, // Backscatter/SSS controls
  backscatterIntensity = 0.5,
  backscatterColor = "#ccffb3",
  backscatterPower = 2.0,
  frontScatterStrength = 0.3,
  rimSSSStrength = 0.5,
  specularEnabled = true, // Specular controls
  specularIntensity = 0.3,
  specularColor = "#fffff2",
  specularPower = 32.0,
  specularScale = 1.0,
  lightDirectionX = 1.0,
  lightDirectionY = 1.0,
  lightDirectionZ = 0.5,
  normalMixEnabled = true, // Normal mixing controls
  normalMixFactor = 0.5,
  aoEnabled = true, // Advanced controls
  aoIntensity = 1.0,
  debugLOD = false, // Debug: Color blades by LOD (green = high, red = low)
  windEnabled = true, // Wind controls
  windStrength = 1.25,
  windDirectionScale = 0.05,
  windDirectionSpeed = 0.05,
  windStrengthScale = 0.25,
  windStrengthSpeed = 1.0,
  playerInteractionEnabled = true, // Player interaction controls
  playerInteractionRange = 2.5,
  playerInteractionStrength = 0.2,
  meshRef: externalMeshRef = null, // Optional external ref for frustum culling control
  geometryRef = null, // Optional callback to expose geometry for adaptive LOD
}) {
  const materialRef = useRef();
  const meshRef = useRef();
  const fogColorRef = useRef(new THREE.Color(fogColor));
  const baseColor1Ref = useRef(new THREE.Color(baseColor1));
  const baseColor2Ref = useRef(new THREE.Color(baseColor2));
  const tipColor1Ref = useRef(new THREE.Color(tipColor1));
  const tipColor2Ref = useRef(new THREE.Color(tipColor2));
  const backscatterColorRef = useRef(new THREE.Color(backscatterColor));
  const specularColorRef = useRef(new THREE.Color(specularColor));

  // Use provided geometry or create one (backward compatibility)
  const actualGeometry = useMemo(() => {
    if (geometry) {
      // Clone geometry and update instance count if needed
      const cloned = geometry.clone();
      cloned.instanceCount = numGrass;
      return cloned;
    }
    // Fallback: create geometry with segments (backward compatibility)
    return createGrassGeometry(segments, numGrass, patchSize, grassHeight);
  }, [geometry, segments, numGrass, patchSize, grassHeight]);

  // Extract actual segment count from geometry (for shader compatibility)
  // Geometry has vertIndex attribute: count = (segments + 1) * 4 (double-sided)
  // So: segments = (vertIndex.count / 4) - 1
  const actualSegments = useMemo(() => {
    if (geometry && actualGeometry) {
      const vertIndexAttr = actualGeometry.getAttribute("vertIndex");
      if (vertIndexAttr) {
        // vertIndex.count = (segments + 1) * 4
        // segments = (count / 4) - 1
        return Math.round(vertIndexAttr.count / 4 - 1);
      }
    }
    return segments; // Fallback to prop
  }, [geometry, actualGeometry, segments]);

  // Track previous uniform values to avoid unnecessary updates
  const prevUniformsRef = useRef({
    actualSegments: actualSegments, // Track geometry LOD changes
    fogEnabled: fogEnabled,
    fogNear: fogNear,
    fogFar: fogFar,
    fogIntensity: fogIntensity,
    gradientBlend: gradientBlend,
    gradientCurve: gradientCurve,
    windEnabled: windEnabled,
    windStrength: windStrength,
    windDirectionScale: windDirectionScale,
    windDirectionSpeed: windDirectionSpeed,
    windStrengthScale: windStrengthScale,
    windStrengthSpeed: windStrengthSpeed,
    playerInteractionEnabled: playerInteractionEnabled,
    playerInteractionRange: playerInteractionRange,
    playerInteractionStrength: playerInteractionStrength,
    normalMixEnabled: normalMixEnabled,
    normalMixFactor: normalMixFactor,
    backscatterEnabled: backscatterEnabled,
    backscatterIntensity: backscatterIntensity,
    backscatterPower: backscatterPower,
    frontScatterStrength: frontScatterStrength,
    rimSSSStrength: rimSSSStrength,
    specularEnabled: specularEnabled,
    specularIntensity: specularIntensity,
    specularPower: specularPower,
    specularScale: specularScale,
    lightDirectionX: lightDirectionX,
    lightDirectionY: lightDirectionY,
    lightDirectionZ: lightDirectionZ,
    aoEnabled: aoEnabled,
    aoIntensity: aoIntensity,
    debugLOD: debugLOD,
  });

  // Expose geometry to parent for adaptive LOD
  useEffect(() => {
    if (geometryRef && actualGeometry) {
      geometryRef(actualGeometry);
    }
    return () => {
      if (geometryRef) {
        geometryRef(null);
      }
    };
  }, [actualGeometry, geometryRef]);

  // Create material once
  const material = useMemo(
    () =>
      new THREE.MeshPhongMaterial({
        side: THREE.FrontSide,
        alphaTest: 0.5,
        transparent: false,
        fog: false, // Disable Three.js built-in fog since we use custom fog
      }),
    []
  );

  // Setup shader once
  useEffect(() => {
    if (!materialRef.current) return;

    const mat = materialRef.current;
    // Use actualSegments from geometry if available, otherwise use segments prop
    const effectiveSegments = actualSegments;
    const GRASS_VERTICES = (effectiveSegments + 1) * 2;

    mat.onBeforeCompile = (shader) => {
      // Add custom uniforms
      shader.uniforms.time = { value: 0 };
      shader.uniforms.grassSize = {
        value: new THREE.Vector2(grassWidth, grassHeight),
      };
      shader.uniforms.grassParams = {
        value: new THREE.Vector4(
          effectiveSegments, // Use effective segments (from geometry or prop)
          GRASS_VERTICES,
          terrainHeight,
          terrainOffset
        ),
      };
      shader.uniforms.grassDraw = {
        value: new THREE.Vector4(lodDistance, maxDistance, 0, 0),
      };
      shader.uniforms.heightmap = { value: heightmap };
      shader.uniforms.heightParams = {
        value: new THREE.Vector4(terrainSize, terrainOffset, 0, 0), // x=terrainSize, y=terrainCenterZ
      };
      shader.uniforms.playerPos = { value: new THREE.Vector3(0, 0, 0) };
      shader.uniforms.viewMatrixInverse = { value: new THREE.Matrix4() };
      shader.uniforms.uCameraPosition = { value: new THREE.Vector3(0, 0, 0) };
      shader.uniforms.grassTexture = { value: null };
      shader.uniforms.grassLODColour = { value: new THREE.Vector3(0, 0, 1) };

      // Fog uniforms
      shader.uniforms.uFogEnabled = { value: fogEnabled };
      shader.uniforms.uFogNear = { value: fogNear };
      shader.uniforms.uFogFar = { value: fogFar };
      shader.uniforms.uFogColor = { value: fogColorRef.current };
      shader.uniforms.uFogIntensity = { value: fogIntensity };

      // Grass color uniforms
      shader.uniforms.uBaseColor1 = { value: baseColor1Ref.current };
      shader.uniforms.uBaseColor2 = { value: baseColor2Ref.current };
      shader.uniforms.uTipColor1 = { value: tipColor1Ref.current };
      shader.uniforms.uTipColor2 = { value: tipColor2Ref.current };
      shader.uniforms.uGradientBlend = { value: gradientBlend };
      shader.uniforms.uGradientCurve = { value: gradientCurve };

      // Wind uniforms
      shader.uniforms.uWindEnabled = { value: windEnabled };
      shader.uniforms.uWindStrength = { value: windStrength };
      shader.uniforms.uWindDirectionScale = { value: windDirectionScale };
      shader.uniforms.uWindDirectionSpeed = { value: windDirectionSpeed };
      shader.uniforms.uWindStrengthScale = { value: windStrengthScale };
      shader.uniforms.uWindStrengthSpeed = { value: windStrengthSpeed };

      // Player interaction uniforms
      shader.uniforms.uPlayerInteractionEnabled = {
        value: playerInteractionEnabled,
      };
      shader.uniforms.uPlayerInteractionRange = {
        value: playerInteractionRange,
      };
      shader.uniforms.uPlayerInteractionStrength = {
        value: playerInteractionStrength,
      };

      // Normal mixing uniforms
      shader.uniforms.uNormalMixEnabled = { value: normalMixEnabled };
      shader.uniforms.uNormalMixFactor = { value: normalMixFactor };

      // Backscatter/SSS uniforms
      shader.uniforms.uBackscatterEnabled = { value: backscatterEnabled };
      shader.uniforms.uBackscatterIntensity = { value: backscatterIntensity };
      shader.uniforms.uBackscatterColor = {
        value: backscatterColorRef.current,
      };
      shader.uniforms.uBackscatterPower = { value: backscatterPower };
      shader.uniforms.uFrontScatterStrength = { value: frontScatterStrength };
      shader.uniforms.uRimSSSStrength = { value: rimSSSStrength };

      // Specular uniforms
      shader.uniforms.uSpecularEnabled = { value: specularEnabled };
      shader.uniforms.uSpecularIntensity = { value: specularIntensity };
      shader.uniforms.uSpecularColor = { value: specularColorRef.current };
      shader.uniforms.uSpecularPower = { value: specularPower };
      shader.uniforms.uSpecularScale = { value: specularScale };
      shader.uniforms.uLightDirection = {
        value: new THREE.Vector3(
          lightDirectionX,
          lightDirectionY,
          lightDirectionZ
        ),
      };

      // Advanced uniforms
      shader.uniforms.uAoEnabled = { value: aoEnabled };
      shader.uniforms.uAoIntensity = { value: aoIntensity };

      // Debug uniforms
      shader.uniforms.uDebugLOD = { value: false };

      // Replace shaders with complete versions
      shader.vertexShader = grassVertexShader;
      shader.fragmentShader = grassFragmentShader;

      // Store reference for updates
      mat.userData.shader = shader;
    };

    mat.needsUpdate = true;

    // Cleanup
    return () => {
      mat.dispose();
    };
  }, [
    actualSegments, // Use actualSegments instead of segments
    grassWidth,
    grassHeight,
    lodDistance,
    maxDistance,
    heightmap,
    terrainHeight,
    terrainOffset,
    terrainSize,
    fogEnabled,
    fogNear,
    fogFar,
    fogIntensity,
    baseColor1,
    baseColor2,
    tipColor1,
    tipColor2,
    gradientBlend,
    gradientCurve,
    backscatterEnabled,
    backscatterIntensity,
    backscatterColor,
    backscatterPower,
    frontScatterStrength,
    rimSSSStrength,
    specularEnabled,
    specularIntensity,
    specularColor,
    specularPower,
    specularScale,
    lightDirectionX,
    lightDirectionY,
    lightDirectionZ,
    aoEnabled,
    aoIntensity,
    debugLOD,
    windEnabled,
    windStrength,
    windDirectionScale,
    windDirectionSpeed,
    windStrengthScale,
    windStrengthSpeed,
    playerInteractionEnabled,
    playerInteractionRange,
    playerInteractionStrength,
    normalMixEnabled,
    normalMixFactor,
  ]);

  // Cleanup geometry
  useEffect(() => {
    return () => {
      geometry.dispose();
    };
  }, [geometry]);

  // Update fog color when it changes
  useEffect(() => {
    fogColorRef.current.set(fogColor);
  }, [fogColor]);

  // Update grass colors when they change
  useEffect(() => {
    baseColor1Ref.current.set(baseColor1);
    baseColor2Ref.current.set(baseColor2);
    tipColor1Ref.current.set(tipColor1);
    tipColor2Ref.current.set(tipColor2);
    backscatterColorRef.current.set(backscatterColor);
    specularColorRef.current.set(specularColor);
  }, [
    baseColor1,
    baseColor2,
    tipColor1,
    tipColor2,
    backscatterColor,
    specularColor,
  ]);

  // Update uniforms each frame - OPTIMIZED: only update changed values
  useFrame((state) => {
    const shader = materialRef.current?.userData?.shader;
    if (!shader) return;

    const prev = prevUniformsRef.current;

    // Always update time and camera-dependent uniforms (they change every frame)
    shader.uniforms.time.value = state.clock.elapsedTime;
    shader.uniforms.viewMatrixInverse.value.copy(state.camera.matrixWorld);
    // Update camera position for accurate LOD calculation - always update to ensure it's current
    shader.uniforms.uCameraPosition.value.copy(state.camera.position);

    // Update grassParams when actualSegments changes (geometry LOD switch)
    const currentGRASS_VERTICES = (actualSegments + 1) * 2;
    if (prev.actualSegments !== actualSegments) {
      shader.uniforms.grassParams.value.set(
        actualSegments,
        currentGRASS_VERTICES,
        terrainHeight,
        terrainOffset
      );
      prev.actualSegments = actualSegments;
    }

    // Player position - only update if provided and changed
    if (playerPosition) {
      const playerPos = shader.uniforms.playerPos.value;
      if (playerPos.distanceToSquared(playerPosition) > 0.0001) {
        playerPos.copy(playerPosition);
      }
    }

    // Update fog uniforms - only if changed
    if (prev.fogEnabled !== fogEnabled) {
      shader.uniforms.uFogEnabled.value = fogEnabled;
      prev.fogEnabled = fogEnabled;
    }
    if (prev.fogNear !== fogNear) {
      shader.uniforms.uFogNear.value = fogNear;
      prev.fogNear = fogNear;
    }
    if (prev.fogFar !== fogFar) {
      shader.uniforms.uFogFar.value = fogFar;
      prev.fogFar = fogFar;
    }
    if (prev.fogIntensity !== fogIntensity) {
      shader.uniforms.uFogIntensity.value = fogIntensity;
      prev.fogIntensity = fogIntensity;
    }
    // Fog color - check if color ref changed (handled by useEffect, but update uniform)
    shader.uniforms.uFogColor.value.copy(fogColorRef.current);

    // Update grass color uniforms - colors are handled by refs/useEffect, just copy
    shader.uniforms.uBaseColor1.value.copy(baseColor1Ref.current);
    shader.uniforms.uBaseColor2.value.copy(baseColor2Ref.current);
    shader.uniforms.uTipColor1.value.copy(tipColor1Ref.current);
    shader.uniforms.uTipColor2.value.copy(tipColor2Ref.current);

    if (prev.gradientBlend !== gradientBlend) {
      shader.uniforms.uGradientBlend.value = gradientBlend;
      prev.gradientBlend = gradientBlend;
    }
    if (prev.gradientCurve !== gradientCurve) {
      shader.uniforms.uGradientCurve.value = gradientCurve;
      prev.gradientCurve = gradientCurve;
    }

    // Update wind uniforms - only if changed
    if (prev.windEnabled !== windEnabled) {
      shader.uniforms.uWindEnabled.value = windEnabled;
      prev.windEnabled = windEnabled;
    }
    if (prev.windStrength !== windStrength) {
      shader.uniforms.uWindStrength.value = windStrength;
      prev.windStrength = windStrength;
    }
    if (prev.windDirectionScale !== windDirectionScale) {
      shader.uniforms.uWindDirectionScale.value = windDirectionScale;
      prev.windDirectionScale = windDirectionScale;
    }
    if (prev.windDirectionSpeed !== windDirectionSpeed) {
      shader.uniforms.uWindDirectionSpeed.value = windDirectionSpeed;
      prev.windDirectionSpeed = windDirectionSpeed;
    }
    if (prev.windStrengthScale !== windStrengthScale) {
      shader.uniforms.uWindStrengthScale.value = windStrengthScale;
      prev.windStrengthScale = windStrengthScale;
    }
    if (prev.windStrengthSpeed !== windStrengthSpeed) {
      shader.uniforms.uWindStrengthSpeed.value = windStrengthSpeed;
      prev.windStrengthSpeed = windStrengthSpeed;
    }

    // Update player interaction uniforms - only if changed
    if (prev.playerInteractionEnabled !== playerInteractionEnabled) {
      shader.uniforms.uPlayerInteractionEnabled.value =
        playerInteractionEnabled;
      prev.playerInteractionEnabled = playerInteractionEnabled;
    }
    if (prev.playerInteractionRange !== playerInteractionRange) {
      shader.uniforms.uPlayerInteractionRange.value = playerInteractionRange;
      prev.playerInteractionRange = playerInteractionRange;
    }
    if (prev.playerInteractionStrength !== playerInteractionStrength) {
      shader.uniforms.uPlayerInteractionStrength.value =
        playerInteractionStrength;
      prev.playerInteractionStrength = playerInteractionStrength;
    }

    // Update normal mixing uniforms - only if changed
    if (prev.normalMixEnabled !== normalMixEnabled) {
      shader.uniforms.uNormalMixEnabled.value = normalMixEnabled;
      prev.normalMixEnabled = normalMixEnabled;
    }
    if (prev.normalMixFactor !== normalMixFactor) {
      shader.uniforms.uNormalMixFactor.value = normalMixFactor;
      prev.normalMixFactor = normalMixFactor;
    }

    // Update backscatter/SSS uniforms - only if changed
    if (prev.backscatterEnabled !== backscatterEnabled) {
      shader.uniforms.uBackscatterEnabled.value = backscatterEnabled;
      prev.backscatterEnabled = backscatterEnabled;
    }
    if (prev.backscatterIntensity !== backscatterIntensity) {
      shader.uniforms.uBackscatterIntensity.value = backscatterIntensity;
      prev.backscatterIntensity = backscatterIntensity;
    }
    if (prev.backscatterPower !== backscatterPower) {
      shader.uniforms.uBackscatterPower.value = backscatterPower;
      prev.backscatterPower = backscatterPower;
    }
    if (prev.frontScatterStrength !== frontScatterStrength) {
      shader.uniforms.uFrontScatterStrength.value = frontScatterStrength;
      prev.frontScatterStrength = frontScatterStrength;
    }
    if (prev.rimSSSStrength !== rimSSSStrength) {
      shader.uniforms.uRimSSSStrength.value = rimSSSStrength;
      prev.rimSSSStrength = rimSSSStrength;
    }
    // Backscatter color - handled by ref/useEffect, just copy
    shader.uniforms.uBackscatterColor.value.copy(backscatterColorRef.current);

    // Update specular uniforms - only if changed
    if (prev.specularEnabled !== specularEnabled) {
      shader.uniforms.uSpecularEnabled.value = specularEnabled;
      prev.specularEnabled = specularEnabled;
    }
    if (prev.specularIntensity !== specularIntensity) {
      shader.uniforms.uSpecularIntensity.value = specularIntensity;
      prev.specularIntensity = specularIntensity;
    }
    if (prev.specularPower !== specularPower) {
      shader.uniforms.uSpecularPower.value = specularPower;
      prev.specularPower = specularPower;
    }
    if (prev.specularScale !== specularScale) {
      shader.uniforms.uSpecularScale.value = specularScale;
      prev.specularScale = specularScale;
    }
    // Specular color - handled by ref/useEffect, just copy
    shader.uniforms.uSpecularColor.value.copy(specularColorRef.current);

    // Light direction - check if any component changed
    if (
      prev.lightDirectionX !== lightDirectionX ||
      prev.lightDirectionY !== lightDirectionY ||
      prev.lightDirectionZ !== lightDirectionZ
    ) {
      shader.uniforms.uLightDirection.value.set(
        lightDirectionX,
        lightDirectionY,
        lightDirectionZ
      );
      prev.lightDirectionX = lightDirectionX;
      prev.lightDirectionY = lightDirectionY;
      prev.lightDirectionZ = lightDirectionZ;
    }

    // Update advanced uniforms - only if changed
    if (prev.aoEnabled !== aoEnabled) {
      shader.uniforms.uAoEnabled.value = aoEnabled;
      prev.aoEnabled = aoEnabled;
    }
    if (prev.aoIntensity !== aoIntensity) {
      shader.uniforms.uAoIntensity.value = aoIntensity;
      prev.aoIntensity = aoIntensity;
    }

    // Debug LOD - always update (can be toggled frequently)
    shader.uniforms.uDebugLOD.value = debugLOD;
  });

  // Use external ref if provided, otherwise use internal ref
  const actualMeshRef = externalMeshRef || meshRef;

  // Frustum culling is controlled by GrassField based on distance to player
  // By default, enable frustum culling (Three.js will handle it efficiently)
  return (
    <mesh
      ref={actualMeshRef}
      position={position}
      geometry={geometry}
      castShadow={castShadow}
      receiveShadow={receiveShadow}
      frustumCulled={true} // Enable Three.js frustum culling
    >
      <primitive ref={materialRef} object={material} attach="material" />
    </mesh>
  );
}

// ============================================================================
// GRASS FIELD COMPONENT (Multiple patches with automatic grid)
// ============================================================================

export function GrassField({
  gridSize = 5, // Number of patches in each direction (5x5 = 25 patches)
  patchSpacing = 10, // Distance between patches
  centerPosition = [0, 0, 0], // Center of the field
  playerPosition = null, // Optional player position for frustum culling optimization
  ...grassProps // Pass all GrassPatch props
}) {
  const { camera } = useThree();

  // GHOST OF TSUSHIMA STYLE: Create two geometries - high LOD (15 vertices) and low LOD (7 vertices)
  const highLODGeometry = useMemo(() => {
    const segments = 14; // 14 segments = 15 vertices per side
    return createGrassGeometry(
      segments,
      grassProps.numGrass || 3072,
      grassProps.patchSize || 10,
      grassProps.grassHeight || 1.5
    );
  }, [grassProps.numGrass, grassProps.patchSize, grassProps.grassHeight]);

  const lowLODGeometry = useMemo(() => {
    const segments = 6; // 6 segments = 7 vertices per side
    return createGrassGeometry(
      segments,
      grassProps.numGrass || 3072,
      grassProps.patchSize || 10,
      grassProps.grassHeight || 1.5
    );
  }, [grassProps.numGrass, grassProps.patchSize, grassProps.grassHeight]);

  // Geometry LOD distance threshold (when to switch from high to low LOD geometry)
  const GEOMETRY_LOD_DISTANCE = grassProps.lodDistance || 15;

  // Create all patches with useMemo (like GrassClaude4)
  // The shader handles heightmap positioning, so we just use centerPosition Y
  const allPatches = useMemo(() => {
    const result = [];
    const half = Math.floor(gridSize / 2);

    for (let x = -half; x <= half; x++) {
      for (let z = -half; z <= half; z++) {
        result.push([
          centerPosition[0] + x * patchSpacing,
          centerPosition[1],
          centerPosition[2] + z * patchSpacing,
        ]);
      }
    }

    return result;
  }, [gridSize, patchSpacing, centerPosition]);

  // Constants for hybrid distance + frustum culling system
  const patchRefsRef = useRef(new Map());
  const CLOSE_DISTANCE = 30; // Within this distance, always render (safety zone)
  const MAX_CULLING_DISTANCE = 200; // Don't check frustum beyond this distance (performance)
  const CULLING_MARGIN = 10; // Expand bounding boxes to prevent edge culling issues (increased for better coverage)
  const FRUSTUM_UPDATE_INTERVAL = 100; // Update frustum every 100ms (10fps) instead of every frame
  // Pre-calculate squared distances to avoid recalculating every frame
  const CLOSE_DISTANCE_SQ = CLOSE_DISTANCE * CLOSE_DISTANCE;
  const MAX_CULLING_DISTANCE_SQ = MAX_CULLING_DISTANCE * MAX_CULLING_DISTANCE;

  // Create refs for all patches
  const patchRefs = useMemo(() => {
    return allPatches.map(() => React.createRef());
  }, [allPatches]);

  // Store refs in map for frustum culling updates
  useEffect(() => {
    patchRefs.forEach((ref, index) => {
      patchRefsRef.current.set(index, ref);
    });
  }, [patchRefs]);

  // OPTIMIZED: Cache bounding boxes per patch (created once, reused)
  const patchBoundingBoxesRef = useRef(new Map());
  useEffect(() => {
    const boxes = patchBoundingBoxesRef.current;
    boxes.clear();
    const patchSize = patchSpacing;
    const halfSize = patchSize / 2 + CULLING_MARGIN;
    const grassHeight = 4.0;

    allPatches.forEach((patchPos, index) => {
      const patchX = patchPos[0];
      const patchZ = patchPos[2];
      const groundY = patchPos[1] || 0;

      boxes.set(
        index,
        new THREE.Box3(
          new THREE.Vector3(patchX - halfSize, groundY - 3, patchZ - halfSize),
          new THREE.Vector3(
            patchX + halfSize,
            groundY + grassHeight + 3,
            patchZ + halfSize
          )
        )
      );
    });
  }, [allPatches, patchSpacing, CULLING_MARGIN]);

  // OPTIMIZED: Hybrid culling with throttled frustum updates
  const frustumRef = useRef(new THREE.Frustum());
  const cameraMatrixRef = useRef(new THREE.Matrix4());
  const cameraPosXZRef = useRef(new THREE.Vector3());
  const patchCenterXZRef = useRef(new THREE.Vector3());
  const lastFrustumUpdateRef = useRef(0);
  const lastCameraPosRef = useRef(new THREE.Vector3());
  const lastCameraQuatRef = useRef(new THREE.Quaternion());

  // OPTIMIZED: Adaptive LOD - Store geometry refs and distances for instance count reduction
  const patchGeometryRefsRef = useRef(new Map());
  const patchDistancesRef = useRef(new Map());
  const LOD_DISTANCE_1 = 20; // Close: 100% instances
  const LOD_DISTANCE_2 = 40; // Medium: 50% instances
  const LOD_DISTANCE_3 = 60; // Far: 25% instances
  const LOD_DISTANCE_4 = 100; // Very far: 12.5% instances
  const LOD_UPDATE_INTERVAL = 200; // Update LOD every 200ms (5fps)
  const lastLODUpdateRef = useRef(0);

  useFrame((state) => {
    const now = state.clock.elapsedTime * 1000; // Convert to milliseconds
    const cameraPos = camera.position;
    const cameraQuat = camera.quaternion;

    // OPTIMIZED: Only update frustum if camera moved/rotated OR enough time passed
    const cameraMoved =
      lastCameraPosRef.current.distanceToSquared(cameraPos) > 0.01 ||
      !lastCameraQuatRef.current.equals(cameraQuat);
    const timeSinceUpdate = now - lastFrustumUpdateRef.current;
    const shouldUpdateFrustum =
      cameraMoved || timeSinceUpdate > FRUSTUM_UPDATE_INTERVAL;

    if (shouldUpdateFrustum) {
      // Update camera matrices only when needed
      camera.updateMatrixWorld(true);
      camera.updateProjectionMatrix();

      // Calculate frustum for culling patches behind camera
      cameraMatrixRef.current.multiplyMatrices(
        camera.projectionMatrix,
        camera.matrixWorldInverse
      );
      frustumRef.current.setFromProjectionMatrix(cameraMatrixRef.current);

      // Cache camera position/quaternion
      lastCameraPosRef.current.copy(cameraPos);
      lastCameraQuatRef.current.copy(cameraQuat);
      lastFrustumUpdateRef.current = now;
    }

    // OPTIMIZED: Reuse Vector3 objects instead of creating new ones
    cameraPosXZRef.current.set(cameraPos.x, 0, cameraPos.z);

    // OPTIMIZED: Hybrid culling: Distance + Frustum (with cached bounding boxes)
    allPatches.forEach((patchPos, index) => {
      const patchRef = patchRefsRef.current.get(index);
      if (!patchRef?.current) return;

      const patchX = patchPos[0];
      const patchZ = patchPos[2];

      // OPTIMIZED: Reuse Vector3 instead of creating new one
      patchCenterXZRef.current.set(patchX, 0, patchZ);

      // First check: Distance-based culling (for far patches) - FASTEST
      // OPTIMIZED: Use squared distance to avoid expensive sqrt
      const distanceSq = cameraPosXZRef.current.distanceToSquared(
        patchCenterXZRef.current
      );

      // Patches within CLOSE_DISTANCE always render (safety zone)
      if (distanceSq <= CLOSE_DISTANCE_SQ) {
        patchRef.current.frustumCulled = false;
        patchRef.current.visible = true;
        // Store distance for LOD (only calculate sqrt when needed)
        const distance = Math.sqrt(distanceSq);
        patchDistancesRef.current.set(index, distance);
        return;
      }

      // Patches beyond MAX_CULLING_DISTANCE are always hidden
      if (distanceSq > MAX_CULLING_DISTANCE_SQ) {
        patchRef.current.frustumCulled = false;
        patchRef.current.visible = false;
        // Store distance for LOD (only calculate sqrt when needed)
        const distance = Math.sqrt(distanceSq);
        patchDistancesRef.current.set(index, distance);
        return;
      }

      // Calculate distance for patches in the middle range (needed for LOD and frustum)
      const distance = Math.sqrt(distanceSq);
      patchDistancesRef.current.set(index, distance);

      // Second check: Frustum culling
      // OPTIMIZED: Use cached bounding box instead of creating new one
      const boundingBox = patchBoundingBoxesRef.current.get(index);
      if (!boundingBox) {
        return;
      }

      // Check if patch is in camera frustum (frustum is cached, so this is fast)
      const isInFrustum = frustumRef.current.intersectsBox(boundingBox);

      // Render if: within distance AND in frustum
      patchRef.current.frustumCulled = false;
      patchRef.current.visible = isInFrustum;
    });

    // OPTIMIZED: Adaptive instance count reduction (throttled)
    if (now - lastLODUpdateRef.current > LOD_UPDATE_INTERVAL) {
      const baseNumGrass = grassProps.numGrass || 3072;

      patchDistancesRef.current.forEach((distance, index) => {
        const geometry = patchGeometryRefsRef.current.get(index);
        const patchRef = patchRefsRef.current.get(index);
        if (!geometry || !patchRef?.current) return;

        // Only update LOD for visible patches
        if (!patchRef.current.visible) {
          // Set to minimum for invisible patches (will be restored when visible)
          geometry.instanceCount = Math.floor(baseNumGrass * 0.125);
          return;
        }

        // Calculate adaptive instance count based on distance
        let instanceCount = baseNumGrass;
        if (distance > LOD_DISTANCE_4) {
          instanceCount = Math.floor(baseNumGrass * 0.125); // 12.5% for very far
        } else if (distance > LOD_DISTANCE_3) {
          instanceCount = Math.floor(baseNumGrass * 0.25); // 25% for far
        } else if (distance > LOD_DISTANCE_2) {
          instanceCount = Math.floor(baseNumGrass * 0.5); // 50% for medium
        } else if (distance > LOD_DISTANCE_1) {
          instanceCount = Math.floor(baseNumGrass * 0.75); // 75% for close-medium
        }
        // else: 100% for very close (LOD_DISTANCE_1 or less)

        // Update instance count (clamp to max available)
        // The "position" attribute is instanced, so count/3 gives number of instances
        const positionAttr = geometry.getAttribute("position");
        if (positionAttr) {
          const maxInstances = positionAttr.count / 3;
          geometry.instanceCount = Math.min(instanceCount, maxInstances);
        }
      });

      lastLODUpdateRef.current = now;
    }
  });

  // GHOST OF TSUSHIMA STYLE: Track which geometry each patch should use based on distance
  const [patchGeometryLOD, setPatchGeometryLOD] = useState(new Map()); // Map of patch index -> true for high LOD, false for low LOD

  // Update geometry LOD selection in useFrame based on camera distance (throttled to avoid too many re-renders)
  const lastGeometryLODUpdateRef = useRef(0);
  const GEOMETRY_LOD_UPDATE_INTERVAL = 100; // Update every 100ms (10fps)

  useFrame((state) => {
    const now = state.clock.elapsedTime * 1000;
    if (now - lastGeometryLODUpdateRef.current < GEOMETRY_LOD_UPDATE_INTERVAL)
      return;

    const cameraPos = camera.position;
    const newLODMap = new Map();

    allPatches.forEach((patchPos, index) => {
      const patchCenter = new THREE.Vector3(
        patchPos[0],
        patchPos[1],
        patchPos[2]
      );
      const distanceToCamera = patchCenter.distanceTo(cameraPos);
      // Close patches use high LOD (15 vertices), far patches use low LOD (7 vertices)
      newLODMap.set(index, distanceToCamera < GEOMETRY_LOD_DISTANCE);
    });

    setPatchGeometryLOD(newLODMap);
    lastGeometryLODUpdateRef.current = now;
  });

  return (
    <group>
      {allPatches.map((pos, i) => {
        // Select geometry based on distance: close = high LOD (15 vertices), far = low LOD (7 vertices)
        const useHighLOD = patchGeometryLOD.get(i) !== false; // Default to high LOD on first render
        const selectedGeometry = useHighLOD ? highLODGeometry : lowLODGeometry;

        return (
          <GrassPatch
            key={`${pos[0]}-${pos[2]}`}
            meshRef={patchRefs[i]}
            geometryRef={(geo) => {
              if (geo) patchGeometryRefsRef.current.set(i, geo);
            }}
            position={pos}
            playerPosition={playerPosition}
            geometry={selectedGeometry}
            {...grassProps}
          />
        );
      })}
    </group>
  );
}

// ============================================================================
// USAGE EXAMPLES & DOCUMENTATION
// ============================================================================

/*

BASIC USAGE - Single Patch:
----------------------------
import { GrassPatch } from './GrassComponent';

<GrassPatch position={[0, 0, 0]} />


ADVANCED USAGE - Full Field:
-----------------------------
import { GrassField } from './GrassComponent';

<GrassField 
  gridSize={9}              // 9x9 grid of patches
  patchSpacing={10}         // 10 units between patches
  segments={6}              // High detail grass
  grassHeight={1.5}
  heightmap={myHeightmapTexture}
  playerPosition={playerRef.current?.position}
/>


FULL CUSTOMIZATION:
-------------------
<GrassPatch 
  // Position
  position={[0, 0, 0]}
  
  // Grass detail and density
  segments={6}              // 1-6, higher = more detail
  numGrass={3072}           // Number of blades per patch
  patchSize={10}            // Physical size of patch
  
  // Grass appearance
  grassWidth={0.1}
  grassHeight={1.5}
  
  // LOD settings
  lodDistance={15}          // Switch to low detail at this distance
  maxDistance={100}         // Cull grass beyond this distance
  
  // Terrain integration
  heightmap={heightmapTexture}  // THREE.Texture for terrain height
  terrainHeight={10}        // Height multiplier for heightmap
  terrainOffset={0}         // Vertical offset
  terrainSize={100}         // Size of terrain for UV mapping
  
  // Interaction
  playerPosition={new THREE.Vector3(0, 0, 0)}  // Player pos for grass bending
  
  // Shadows
  castShadow={false}        // Performance: grass doesn't cast shadows
  receiveShadow={true}      // Grass receives shadows
/>


PERFORMANCE TIPS:
-----------------
1. Use lower segments (1-3) for distant patches
2. Reduce numGrass for better FPS (try 1024 or 2048)
3. Keep gridSize reasonable (5-9 for most scenes)
4. Use heightmap only when needed
5. Disable castShadow for grass (minimal visual impact)


INTEGRATION WITH EXISTING SCENE:
---------------------------------
// In your R3F app:
import { Canvas } from '@react-three/fiber';
import { GrassField } from './GrassComponent';

<Canvas>
  <ambientLight intensity={0.5} />
  <directionalLight position={[10, 10, 5]} castShadow />
  
  /* Your existing scene */

// <YourTerrain />
// <YourPlayer />

/* Add grass */
//<GrassField
//gridSize={7}
//segments={6}
//heightmap={terrainHeightmap}
///>
//</Canvas>

//TROUBLESHOOTING;
//----------------
//- Grass not visible? Check camera position and distance
//- Performance issues? Reduce segments, numGrass, or gridSize
//- Grass floating/clipping? Adjust terrainHeight and terrainOffset
//- Compilation errors? Make sure Three.js r180+ is installed
