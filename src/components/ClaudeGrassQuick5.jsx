import React, { useRef, useMemo, useEffect } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";

// Constants
const NUM_GRASS = 32 * 32 * 3; // 3072 blades per patch
const GRASS_SEGMENTS_LOW = 1;
const GRASS_SEGMENTS_HIGH = 6;
const GRASS_VERTICES_LOW = (GRASS_SEGMENTS_LOW + 1) * 2; // 4
const GRASS_VERTICES_HIGH = (GRASS_SEGMENTS_HIGH + 1) * 2; // 14
const GRASS_LOD_DIST = 15;
const GRASS_MAX_DIST = 100;
const GRASS_PATCH_SIZE = 5 * 2; // 10 units
const GRASS_WIDTH = 0.1;
const GRASS_HEIGHT = 1.5;

// Shader utility functions that will be prepended to both shaders
// Note: PI is already defined by Three.js <common>, so we don't define it here
const SHADER_COMMON = `
// Utility functions
float saturate(float x) {
  return clamp(x, 0.0, 1.0);
}

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
        1.0, 0.0, 0.0,
        0.0, c, -s,
        0.0, s, c
    );
}

mat3 rotateY(float theta) {
    float c = cos(theta);
    float s = sin(theta);
    return mat3(
        c, 0.0, s,
        0.0, 1.0, 0.0,
        -s, 0.0, c
    );
}

mat3 rotateAxis(vec3 axis, float angle) {
  axis = normalize(axis);
  float s = sin(angle);
  float c = cos(angle);
  float oc = 1.0 - c;

  return mat3(
    oc * axis.x * axis.x + c, oc * axis.x * axis.y + axis.z * s, oc * axis.z * axis.x - axis.y * s,
    oc * axis.x * axis.y - axis.z * s, oc * axis.y * axis.y + c, oc * axis.y * axis.z + axis.x * s,
    oc * axis.z * axis.x + axis.y * s, oc * axis.y * axis.z - axis.x * s, oc * axis.z * axis.z + c
  );
}

// Hash functions - GLSL 1.0 compatible (no uint types)
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

float hash12(vec2 p) {
  return hash(p);
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

// OkLab color space
vec3 rgbToOklab(vec3 c) {
  mat3 kCONEtoLMS = mat3(
    0.4121656120, 0.5362752080, 0.0514575653,
    0.2118591070, 0.6807189584, 0.1074065790,
    0.0883097947, 0.2818474174, 0.6302613616
  );

  vec3 lms = kCONEtoLMS * c;
  return sign(lms) * pow(abs(lms), vec3(0.3333333333333));
}

vec3 oklabToRGB(vec3 c) {
  mat3 kLMStoCONE = mat3(
    4.0767245293, -3.3072168827, 0.2307590544,
    -1.2681437731, 2.6093323231, -0.3411344290,
    -0.0041119885, -0.7034763098, 1.7068625689
  );

  vec3 lms = c;
  return kLMStoCONE * (lms * lms * lms);
}

vec3 col3_rgb(float r, float g, float b) {
  return rgbToOklab(vec3(r, g, b));
}

vec3 col3_vec(vec3 v) {
  return rgbToOklab(v);
}

// Sky/Fog functions
vec3 CalculateSkyLighting(vec3 viewDir, vec3 normalDir) {
  vec3 SKY_lighterBlue = vec3(0.39, 0.57, 0.86) * 0.25;
  vec3 SKY_midBlue = vec3(0.1, 0.11, 0.1) * 0.5;
  vec3 SKY_darkerBlue = vec3(0.0);
  vec3 SKY_SUN_COLOUR = vec3(0.5);
  vec3 SKY_SUN_GLOW_COLOUR = vec3(0.15, 0.2, 0.25);
  vec3 SUN_DIR = vec3(-1.0, 0.45, 1.0);

  vec3 lighterBlue = col3_vec(SKY_lighterBlue);
  vec3 midBlue = col3_vec(SKY_midBlue);
  vec3 darkerBlue = col3_vec(SKY_darkerBlue);
  vec3 SUN_COLOUR = col3_vec(SKY_SUN_COLOUR);
  vec3 SUN_GLOW_COLOUR = col3_vec(SKY_SUN_GLOW_COLOUR);

  float viewDirY = linearstep(-0.01, 1.0, viewDir.y);
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
`;

// Vertex Shader
const vertexShader = `
#define PHONG
varying vec3 vViewPosition;

${SHADER_COMMON}

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

varying vec3 vWorldNormal;
varying vec3 vGrassColour;
varying vec4 vGrassParams;
varying vec3 vNormal2;
varying vec3 vWorldPosition;

uniform vec2 grassSize;
uniform vec4 grassParams;
uniform vec4 grassDraw;
uniform float time;
uniform sampler2D heightmap;
uniform vec4 heightParams;
uniform vec3 playerPos;
uniform mat4 viewMatrixInverse;
uniform vec4 windParams; // x: dirScale, y: dirSpeed, z: strengthScale, w: strengthSpeed
uniform float windStrength;
uniform vec2 playerInteractionParams; // x: range, y: strength
uniform float playerInteractionRepel; // 1.0 for repel, -1.0 for attract
uniform float playerInteractionHeightThreshold;
uniform vec3 uBaseColor1;
uniform vec3 uBaseColor2;
uniform vec3 uTipColor1;
uniform vec3 uTipColor2;
uniform float uGradientCurve;
uniform bool uAoEnabled;
uniform float uAoIntensity;

attribute float vertIndex;

void main() {
  #include <uv_vertex>
  #include <color_vertex>
  #include <morphcolor_vertex>
  #include <beginnormal_vertex>
  #include <begin_vertex>
  
  // Declare variables inline as we use them (like GrassClaude2)
  vec3 grassOffset;
  vec3 grassBladeWorldPos;
  vec2 heightmapUV;
  vec4 heightmapSample;
  float heightmapSampleHeight;
  vec4 hashVal1;
  float highLODOut;
  float lodFadeIn;
  float isSandy;
  float grassAllowedHash;
  float isGrassAllowed;
  float randomAngle;
  float randomShade;
  float randomHeight;
  float randomWidth;
  float randomLean;
  vec2 hashGrassColour;
  float leanAnimation;
  float GRASS_SEGMENTS;
  float GRASS_VERTICES;
  float vertID;
  float zSide;
  float xSide;
  float heightPercent;
  float grassTotalHeight;
  float grassTotalWidthHigh;
  float grassTotalWidthLow;
  float grassTotalWidth;
  float x;
  float y;
  float windDir;
  float windNoiseSample;
  float windLeanAngle;
  vec3 windAxis;
  float distToPlayer;
  float heightDiff;
  float heightFalloff;
  float distanceFalloff;
  float playerFalloff;
  float playerLeanAngle;
  vec3 grassToPlayer;
  vec3 playerLeanAxis;
  float easedHeight;
  float curveAmount;
  float ncurve1;
  vec3 n1;
  float ncurve2;
  vec3 n2;
  vec3 ncurve;
  mat3 grassMat;
  vec3 grassFaceNormal;
  vec3 grassVertexNormal;
  vec3 grassVertexNormal1;
  vec3 grassVertexNormal2;
  vec3 grassVertexPosition;
  vec3 b1;
  vec3 b2;
  vec3 t1;
  vec3 t2;
  vec3 baseColour;
  vec3 tipColour;
  vec3 highLODColour;
  vec3 lowLODColour;
  float SKY_RATIO;
  vec3 UP;
  float skyFadeIn;
  vec3 normal1;
  vec3 normal2;
  // transformed is declared by <begin_vertex>, don't redeclare it
  vec3 viewDir;
  vec3 viewDirXZ;
  vec3 grassFaceNormalXZ;
  float viewDotNormal;
  float viewSpaceThickenFactor;
  
  // Now execute the code
  grassOffset = vec3(position.x, 0.0, position.y);

  // Blade world position
  grassBladeWorldPos = (modelMatrix * vec4(grassOffset, 1.0)).xyz;
  heightmapUV = vec2(
      remap(grassBladeWorldPos.x, -heightParams.x * 0.5, heightParams.x * 0.5, 0.0, 1.0),
      remap(grassBladeWorldPos.z, -heightParams.x * 0.5, heightParams.x * 0.5, 1.0, 0.0));
  heightmapSample = texture2D(heightmap, heightmapUV);
  grassBladeWorldPos.y += heightmapSample.x * grassParams.z - grassParams.w;

  heightmapSampleHeight = 1.0;

  hashVal1 = hash42(vec2(grassBladeWorldPos.x, grassBladeWorldPos.z));

  highLODOut = smoothstep(grassDraw.x * 0.5, grassDraw.x, distance(cameraPosition, grassBladeWorldPos));
  lodFadeIn = smoothstep(grassDraw.x, grassDraw.y, distance(cameraPosition, grassBladeWorldPos));

  // Check terrain type
  isSandy = linearstep(-11.0, -14.0, grassBladeWorldPos.y);
  grassAllowedHash = hashVal1.w - isSandy;
  isGrassAllowed = step(0.0, grassAllowedHash);

  randomAngle = hashVal1.x * 2.0 * PI;
  randomShade = remap(hashVal1.y, -1.0, 1.0, 0.5, 1.0);
  randomHeight = remap(hashVal1.z, 0.0, 1.0, 0.75, 1.5) * mix(1.0, 0.0, lodFadeIn) * isGrassAllowed * heightmapSampleHeight;
  randomWidth = (1.0 - isSandy) * heightmapSampleHeight;
  randomLean = remap(hashVal1.w, 0.0, 1.0, 0.1, 0.4);

  hashGrassColour = hash22(vec2(grassBladeWorldPos.x, grassBladeWorldPos.z));
  leanAnimation = noise12(vec2(time * 0.35) + grassBladeWorldPos.xz * 137.423) * 0.1;

  GRASS_SEGMENTS = grassParams.x;
  GRASS_VERTICES = grassParams.y;

  // Figure out vertex id
  vertID = mod(float(vertIndex), GRASS_VERTICES);

  // 1 = front, -1 = back
  zSide = -(floor(vertIndex / GRASS_VERTICES) * 2.0 - 1.0);

  // 0 = left, 1 = right
  xSide = mod(vertID, 2.0);

  heightPercent = (vertID - xSide) / (GRASS_SEGMENTS * 2.0);

  grassTotalHeight = grassSize.y * randomHeight;
  grassTotalWidthHigh = easeOut(1.0 - heightPercent, 2.0);
  grassTotalWidthLow = 1.0 - heightPercent;
  grassTotalWidth = grassSize.x * mix(grassTotalWidthHigh, grassTotalWidthLow, highLODOut) * randomWidth;

  // Shift verts
  x = (xSide - 0.5) * grassTotalWidth;
  y = heightPercent * grassTotalHeight;

  // Wind - using uniforms
  windDir = noise12(grassBladeWorldPos.xz * windParams.x + windParams.y * time);
  windNoiseSample = noise12(grassBladeWorldPos.xz * windParams.z + time * windParams.w);
  windLeanAngle = remap(windNoiseSample, -1.0, 1.0, 0.25, 1.0);
  windLeanAngle = easeIn(windLeanAngle, 2.0) * windStrength;
  windAxis = vec3(cos(windDir), 0.0, sin(windDir));
  windLeanAngle *= heightPercent;

  // Player interaction - using uniforms
  distToPlayer = distance(grassBladeWorldPos.xz, playerPos.xz);
  heightDiff = abs(grassBladeWorldPos.y - playerPos.y);
  // Only affect grass if player is within reasonable height range
  heightFalloff = smoothstep(playerInteractionHeightThreshold, 0.0, heightDiff);
  // Distance falloff (horizontal only)
  distanceFalloff = smoothstep(playerInteractionParams.x, 1.0, distToPlayer);
  // Combine both falloffs - player must be both close horizontally AND at ground level
  playerFalloff = distanceFalloff * heightFalloff;
  playerLeanAngle = mix(0.0, playerInteractionParams.y, playerFalloff * linearstep(0.5, 0.0, windLeanAngle));
  // Apply repel/attract: repel = positive angle (bend away), attract = negative angle (bend toward)
  playerLeanAngle *= playerInteractionRepel;
  grassToPlayer = normalize(vec3(playerPos.x, 0.0, playerPos.z) - vec3(grassBladeWorldPos.x, 0.0, grassBladeWorldPos.z));
  playerLeanAxis = vec3(grassToPlayer.z, 0.0, -grassToPlayer.x);

  randomLean += leanAnimation;

  easedHeight = mix(easeIn(heightPercent, 2.0), 1.0, highLODOut);
  curveAmount = -randomLean * easedHeight;

  // Normal calculation
  ncurve1 = -randomLean * easedHeight;
  n1 = vec3(0.0, (heightPercent + 0.01), 0.0);
  n1 = rotateX(ncurve1) * n1;

  ncurve2 = -randomLean * easedHeight * 0.9;
  n2 = vec3(0.0, (heightPercent + 0.01) * 0.9, 0.0);
  n2 = rotateX(ncurve2) * n2;

  ncurve = normalize(n1 - n2);

  grassMat = rotateAxis(playerLeanAxis, playerLeanAngle) * rotateAxis(windAxis, windLeanAngle) * rotateY(randomAngle);

  grassFaceNormal = vec3(0.0, 0.0, 1.0);
  grassFaceNormal = grassMat * grassFaceNormal;
  grassFaceNormal *= zSide;

  grassVertexNormal = vec3(0.0, -ncurve.z, ncurve.y);
  grassVertexNormal1 = rotateY(PI * 0.3 * zSide) * grassVertexNormal;
  grassVertexNormal2 = rotateY(PI * -0.3 * zSide) * grassVertexNormal;

  grassVertexNormal1 = grassMat * grassVertexNormal1;
  grassVertexNormal1 *= zSide;

  grassVertexNormal2 = grassMat * grassVertexNormal2;
  grassVertexNormal2 *= zSide;

  grassVertexPosition = vec3(x, y, 0.0);
  grassVertexPosition = rotateX(curveAmount) * grassVertexPosition;
  grassVertexPosition = grassMat * grassVertexPosition;
  grassVertexPosition += grassOffset;

  // Color gradient - using uniforms
  b1 = uBaseColor1;
  b2 = uBaseColor2;
  t1 = uTipColor1;
  t2 = uTipColor2;

  baseColour = mix(b1, b2, hashGrassColour.x);
  tipColour = mix(t1, t2, hashGrassColour.y);
  highLODColour = mix(baseColour, tipColour, easeIn(heightPercent, uGradientCurve)) * randomShade;
  lowLODColour = mix(b1, t1, heightPercent);
  vGrassColour = mix(highLODColour, lowLODColour, highLODOut);
  vGrassParams = vec4(heightPercent, grassBladeWorldPos.y, highLODOut, xSide);

  SKY_RATIO = 0.25;
  UP = vec3(0.0, 1.0, 0.0);
  skyFadeIn = (1.0 - highLODOut) * SKY_RATIO;
  normal1 = normalize(mix(UP, grassVertexNormal1, skyFadeIn));
  normal2 = normalize(mix(UP, grassVertexNormal2, skyFadeIn));

  transformed = grassVertexPosition;
  transformed.y += grassBladeWorldPos.y;

  viewDir = normalize(cameraPosition - grassBladeWorldPos);
  viewDirXZ = normalize(vec3(viewDir.x, 0.0, viewDir.z));
  grassFaceNormalXZ = normalize(vec3(grassFaceNormal.x, 0.0, grassFaceNormal.z));

  viewDotNormal = saturate(dot(grassFaceNormal, viewDirXZ));
  viewSpaceThickenFactor = easeOut(1.0 - viewDotNormal, 4.0) * smoothstep(0.0, 0.2, viewDotNormal);

  // Set objectNormal before normal processing includes
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

  // Billboard thickening in view space
  mvPosition.x += viewSpaceThickenFactor * (xSide - 0.5) * grassTotalWidth * 0.5 * zSide;

  gl_Position = projectionMatrix * mvPosition;

  #include <logdepthbuf_vertex>
  #include <clipping_planes_vertex>
  vViewPosition = -mvPosition.xyz;
  #include <worldpos_vertex>
  #include <envmap_vertex>
  #include <shadowmap_vertex>
  #include <fog_vertex>

  // Calculate world position manually (worldPosition from worldpos_vertex might not be accessible)
  vWorldPosition = (modelMatrix * vec4(transformed, 1.0)).xyz;
}
`;

// Fragment Shader
const fragmentShader = `
#define PHONG
uniform vec3 diffuse;
uniform vec3 emissive;
uniform vec3 specular;
uniform float shininess;
uniform float opacity;

#include <common>

varying vec3 vGrassColour;
varying vec4 vGrassParams;
varying vec3 vNormal2;
varying vec3 vWorldPosition;
// vViewPosition is provided by Three.js shader chunks, don't declare it

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

uniform sampler2D grassTexture;
uniform vec3 grassLODColour;
uniform float time;
uniform mat3 normalMatrix;
uniform bool uAoEnabled;
uniform float uAoIntensity;
uniform bool uFogEnabled;
uniform float uFogNear;
uniform float uFogFar;
uniform float uFogIntensity;
uniform vec3 uFogColor;
uniform bool uSpecularEnabled;
uniform float uSpecularIntensity;
uniform vec3 uSpecularColor;
uniform vec3 uSpecularDirection;
uniform float uGrassMiddleBrightnessMin;
uniform float uGrassMiddleBrightnessMax;
uniform bool uBackscatterEnabled;
uniform float uBackscatterIntensity;
uniform vec3 uBackscatterColor;
uniform float uBackscatterPower;
uniform float uFrontScatterStrength;
uniform float uRimSSSStrength;

// Utility functions
// Note: saturate might be defined by Three.js, so we check and undefine if needed
#ifdef saturate
#undef saturate
#endif
float saturate(float x) {
  return clamp(x, 0.0, 1.0);
}

float linearstep(float edge0, float edge1, float x) {
  return clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
}

float easeIn(float t, float p) {
  return pow(t, p);
}

// OkLab color space functions for fog
vec3 rgbToOklab(vec3 c) {
  mat3 kCONEtoLMS = mat3(
    0.4121656120, 0.5362752080, 0.0514575653,
    0.2118591070, 0.6807189584, 0.1074065790,
    0.0883097947, 0.2818474174, 0.6302613616
  );
  vec3 lms = kCONEtoLMS * c;
  return sign(lms) * pow(abs(lms), vec3(0.3333333333333));
}

vec3 oklabToRGB(vec3 c) {
  mat3 kLMStoCONE = mat3(
    4.0767245293, -3.3072168827, 0.2307590544,
    -1.2681437731, 2.6093323231, -0.3411344290,
    -0.0041119885, -0.7034763098, 1.7068625689
  );
  vec3 lms = c;
  return kLMStoCONE * (lms * lms * lms);
}

vec3 col3_vec(vec3 v) {
  return rgbToOklab(v);
}

// Sky/Fog functions
vec3 CalculateSkyLighting(vec3 viewDir, vec3 normalDir) {
  vec3 SKY_lighterBlue = vec3(0.39, 0.57, 0.86) * 0.25;
  vec3 SKY_midBlue = vec3(0.1, 0.11, 0.1) * 0.5;
  vec3 SKY_darkerBlue = vec3(0.0);
  vec3 SKY_SUN_COLOUR = vec3(0.5);
  vec3 SKY_SUN_GLOW_COLOUR = vec3(0.15, 0.2, 0.25);
  vec3 SUN_DIR = vec3(-1.0, 0.45, 1.0);

  vec3 lighterBlue = col3_vec(SKY_lighterBlue);
  vec3 midBlue = col3_vec(SKY_midBlue);
  vec3 darkerBlue = col3_vec(SKY_darkerBlue);
  vec3 SUN_COLOUR = col3_vec(SKY_SUN_COLOUR);
  vec3 SUN_GLOW_COLOUR = col3_vec(SKY_SUN_GLOW_COLOUR);

  float viewDirY = linearstep(-0.01, 1.0, viewDir.y);
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

// CalculateFog is only used in fragment shader, so it's defined here
// (not in SHADER_COMMON) to avoid needing fog uniforms in vertex shader
vec3 CalculateFog(vec3 baseColour, vec3 viewDir, float sceneDepth) {
  if (!uFogEnabled) {
    return baseColour;
  }
  
  // Use linear depth for more visible and controllable fog
  float fogDepth = sceneDepth;
  
  // Calculate fog factor (0 = no fog, 1 = full fog)
  float fogFactor = clamp((fogDepth - uFogNear) / (uFogFar - uFogNear), 0.0, 1.0);
  
  // Apply fog intensity
  fogFactor *= uFogIntensity;
  fogFactor = clamp(fogFactor, 0.0, 1.0);

  // Mix base color with fog color
  vec3 finalColour = mix(baseColour, uFogColor, fogFactor);
  return finalColour;
}

void main() {
  // Declare ALL variables at the top
  vec3 viewDir;
  vec4 diffuseColor;
  float heightPercent;
  float height;
  float lodFadeIn;
  float lodFadeOut;
  float grassMiddle;
  float isSandy;
  float density;
  float aoForDensity;
  float ao;
  ReflectedLight reflectedLight;
  vec3 totalEmissiveRadiance;
  vec3 normal2;
  // baseNormal will be declared inline where it's used
  vec3 outgoingLight;
  float sceneDepth;
  
  // Now execute code
  viewDir = normalize(cameraPosition - vWorldPosition);

  #include <clipping_planes_fragment>
  
  diffuseColor = vec4(diffuse, opacity);

  // Grass
  heightPercent = vGrassParams.x;
  height = vGrassParams.y;
  lodFadeIn = vGrassParams.z;
  lodFadeOut = 1.0 - lodFadeIn;

  grassMiddle = mix(smoothstep(abs(vGrassParams.w - 0.5), 0.0, 0.1), 1.0, lodFadeIn);

  isSandy = saturate(linearstep(-11.0, -14.0, height));

  density = 1.0 - isSandy;

  // Ambient Occlusion - darker at base, brighter at tip
  if (uAoEnabled) {
    aoForDensity = mix(1.0, 0.25, density);
    ao = mix(aoForDensity, 1.0, easeIn(heightPercent, 2.0));
    diffuseColor.rgb *= ao * uAoIntensity;
  }

  diffuseColor.rgb *= vGrassColour;
  diffuseColor.rgb *= mix(uGrassMiddleBrightnessMin, uGrassMiddleBrightnessMax, grassMiddle);

  reflectedLight = ReflectedLight(vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0));
  totalEmissiveRadiance = emissive;
  
  #include <logdepthbuf_fragment>
  #include <map_fragment>
  #include <color_fragment>
  #include <alphamap_fragment>
  #include <alphatest_fragment>
  #include <alphahash_fragment>
  #include <specularmap_fragment>
  #include <normal_fragment_begin>
  #include <normal_fragment_maps>

  // Mix normals for more 3D appearance - using normal (from Three.js) and vNormal2 (custom)
  // normal is already provided by Three.js after normal_fragment_maps
  // Use a different variable name to avoid conflicts with Three.js
  vec3 grassBaseNormal = normalize(normal);
  normal2 = normalize(vNormal2);
  normal = normalize(mix(grassBaseNormal, normal2, vGrassParams.w));

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
  
  // Specular highlight (moon reflection style)
  if (uSpecularEnabled && uSpecularIntensity > 0.0) {
    // Use vertex normal for consistent specular across grass blades
    vec3 specularNormal = normalize(vNormal2);
    vec3 viewDir = normalize(-vViewPosition);
    // Specular direction in world space
    vec3 specDir = normalize(uSpecularDirection);
    // Reflect specular direction off normal
    vec3 specReflectDir = reflect(-specDir, specularNormal);
    // Calculate specular highlight (using same power as simonDevGrass22: 25.6)
    float spec = pow(max(dot(viewDir, specReflectDir), 0.0), 25.6);
    
    // Distance-based falloff - reduce specular when camera is close to prevent red glow on nearby grass
    float specularDepth = length(vViewPosition);
    float distanceFalloff = smoothstep(2.0, 10.0, specularDepth);
    // Also add tip falloff - specular should be stronger at tips
    float tipFalloff = smoothstep(0.5, 1.0, heightPercent);
    
    // Apply color, intensity, distance falloff, tip falloff, and 3x multiplier for visibility
    vec3 specular = uSpecularColor * spec * uSpecularIntensity * distanceFalloff * tipFalloff * 3.0;
    reflectedLight.directSpecular += specular;
  }
  
  #include <aomap_fragment>
  
  outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + reflectedLight.directSpecular + reflectedLight.indirectSpecular + totalEmissiveRadiance;

  #include <envmap_fragment>
  #include <opaque_fragment>
  #include <tonemapping_fragment>
  #include <colorspace_fragment>

  // Calculate fog with proper depth
  sceneDepth = length(vViewPosition);
  gl_FragColor.xyz = CalculateFog(gl_FragColor.xyz, viewDir, sceneDepth);

  #include <premultiplied_alpha_fragment>
  #include <dithering_fragment>
}
`;

// Custom Float16 Buffer Attribute (for memory optimization)
class InstancedFloat16BufferAttribute extends THREE.InstancedBufferAttribute {
  constructor(array, itemSize, normalized = false, meshPerAttribute = 1) {
    super(new Uint16Array(array), itemSize, normalized, meshPerAttribute);
    this.isFloat16BufferAttribute = true;
  }
}

// Seeded random for consistent grass placement
let seed = 0;
function setSeed(s) {
  seed = s;
}

function random() {
  const x = Math.sin(seed++) * 10000;
  return x - Math.floor(x);
}

function randRange(min, max) {
  return min + random() * (max - min);
}

// Create geometry for grass
function createGrassGeometry(segments, numGrass) {
  setSeed(0);

  const VERTICES = (segments + 1) * 2;

  // Create indices for front and back faces
  const indices = [];
  for (let i = 0; i < segments; ++i) {
    const vi = i * 2;
    // Front face
    indices.push(vi + 0, vi + 1, vi + 2);
    indices.push(vi + 2, vi + 1, vi + 3);

    // Back face
    const fi = VERTICES + vi;
    indices.push(fi + 2, fi + 1, fi + 0);
    indices.push(fi + 3, fi + 1, fi + 2);
  }

  // Create random offsets for each blade within patch
  const offsets = [];
  for (let i = 0; i < numGrass; ++i) {
    offsets.push(randRange(-GRASS_PATCH_SIZE * 0.5, GRASS_PATCH_SIZE * 0.5));
    offsets.push(randRange(-GRASS_PATCH_SIZE * 0.5, GRASS_PATCH_SIZE * 0.5));
    offsets.push(0);
  }

  // Convert to Float16 for memory efficiency
  const offsetsData = offsets.map(THREE.DataUtils.toHalfFloat);

  // Vertex indices
  const vertID = new Uint8Array(VERTICES * 2);
  for (let i = 0; i < VERTICES * 2; ++i) {
    vertID[i] = i;
  }

  const geo = new THREE.InstancedBufferGeometry();
  geo.instanceCount = numGrass;
  geo.setAttribute("vertIndex", new THREE.Uint8BufferAttribute(vertID, 1));
  geo.setAttribute(
    "position",
    new InstancedFloat16BufferAttribute(offsetsData, 3)
  );
  geo.setIndex(indices);
  geo.boundingSphere = new THREE.Sphere(
    new THREE.Vector3(0, 0, 0),
    1 + GRASS_PATCH_SIZE * 2
  );

  return geo;
}

// Create heightmap texture (simple flat plane for now)
function createHeightmap() {
  const size = 256;
  const data = new Uint8Array(size * size * 4);

  for (let i = 0; i < size * size; i++) {
    const stride = i * 4;
    data[stride] = 0; // Height
    data[stride + 1] = 0;
    data[stride + 2] = 0;
    data[stride + 3] = 255;
  }

  const texture = new THREE.DataTexture(data, size, size, THREE.RGBAFormat);
  texture.needsUpdate = true;

  return texture;
}

// Main component
export default function ClaudeGrassQuick5({
  playerPosition = new THREE.Vector3(0, 0, 0),
  terrainSize = 100,
  heightScale = 1,
  heightOffset = 0,
  grassWidth = 0.1,
  grassHeight = 1.5,
  lodDistance = 15,
  maxDistance = 100,
  patchSize = 10,
  gridSize = 16,
  patchSpacing = 10,
  windEnabled = true,
  windStrength = 1.25,
  windDirectionScale = 0.05,
  windDirectionSpeed = 0.05,
  windStrengthScale = 0.25,
  windStrengthSpeed = 1.0,
  playerInteractionEnabled = true,
  playerInteractionRepel = true,
  playerInteractionRange = 2.5,
  playerInteractionStrength = 0.2,
  playerInteractionHeightThreshold = 3.0,
  baseColor1 = "#051303",
  baseColor2 = "#061a03",
  tipColor1 = "#a6cc40",
  tipColor2 = "#cce666",
  gradientCurve = 4.0,
  aoEnabled = true,
  aoIntensity = 1.0,
  grassMiddleBrightnessMin = 0.85,
  grassMiddleBrightnessMax = 1.0,
  fogEnabled = false,
  fogNear = 5.0,
  fogFar = 50.0,
  fogIntensity = 1.0,
  fogColor = "#4f74af",
  specularEnabled = false,
  specularIntensity = 2.0,
  specularColor = "#ffffff",
  specularDirectionX = -1.0,
  specularDirectionY = 1.0,
  specularDirectionZ = 0.5,
  backscatterEnabled = true,
  backscatterIntensity = 0.5,
  backscatterColor = "#51cc66",
  backscatterPower = 2.0,
  frontScatterStrength = 0.3,
  rimSSSStrength = 0.5,
  grassDensity = 3072, // Number of grass blades per patch (default: 32*32*3)
}) {
  const groupRef = useRef();
  const { camera } = useThree();

  // Helper function to convert sRGB to linear
  const convertSRGBToLinear = (color) => {
    const c = new THREE.Color(color);
    // Convert each channel from sRGB to linear
    c.r = c.r <= 0.04045 ? c.r / 12.92 : Math.pow((c.r + 0.055) / 1.055, 2.4);
    c.g = c.g <= 0.04045 ? c.g / 12.92 : Math.pow((c.g + 0.055) / 1.055, 2.4);
    c.b = c.b <= 0.04045 ? c.b / 12.92 : Math.pow((c.b + 0.055) / 1.055, 2.4);
    return c;
  };

  // Color refs for uniform updates - convert sRGB to linear
  const baseColor1Ref = useRef(convertSRGBToLinear(baseColor1));
  const baseColor2Ref = useRef(convertSRGBToLinear(baseColor2));
  const tipColor1Ref = useRef(convertSRGBToLinear(tipColor1));
  const tipColor2Ref = useRef(convertSRGBToLinear(tipColor2));
  const fogColorRef = useRef(convertSRGBToLinear(fogColor));
  const specularColorRef = useRef(convertSRGBToLinear(specularColor));
  const backscatterColorRef = useRef(convertSRGBToLinear(backscatterColor));

  // Update color refs when props change - convert to linear space
  useEffect(() => {
    const c1 = convertSRGBToLinear(baseColor1);
    const c2 = convertSRGBToLinear(baseColor2);
    const c3 = convertSRGBToLinear(tipColor1);
    const c4 = convertSRGBToLinear(tipColor2);
    const fog = convertSRGBToLinear(fogColor);
    const spec = convertSRGBToLinear(specularColor);
    baseColor1Ref.current.copy(c1);
    baseColor2Ref.current.copy(c2);
    tipColor1Ref.current.copy(c3);
    tipColor2Ref.current.copy(c4);
    fogColorRef.current.copy(fog);
    specularColorRef.current.copy(spec);
    const backscatter = convertSRGBToLinear(backscatterColor);
    backscatterColorRef.current.copy(backscatter);
  }, [
    baseColor1,
    baseColor2,
    tipColor1,
    tipColor2,
    fogColor,
    specularColor,
    backscatterColor,
  ]);

  // Create geometries and materials - recreate when grassDensity changes
  const { geometryLow, geometryHigh, materialLow, materialHigh, heightmap } =
    useMemo(() => {
      const geoLow = createGrassGeometry(GRASS_SEGMENTS_LOW, grassDensity);
      const geoHigh = createGrassGeometry(GRASS_SEGMENTS_HIGH, grassDensity);
      const hmap = createHeightmap();

      const createMaterial = (segments, vertices) => {
        const mat = new THREE.MeshPhongMaterial({
          color: 0xffffff, // White - grass colors come from uniforms, not material color
          side: THREE.FrontSide,
        });

        mat.onBeforeCompile = (shader) => {
          shader.uniforms.grassSize = {
            value: new THREE.Vector2(GRASS_WIDTH, GRASS_HEIGHT),
          };
          shader.uniforms.grassParams = {
            value: new THREE.Vector4(
              segments,
              vertices,
              1, // heightScale - will be updated via useEffect
              0 // heightOffset - will be updated via useEffect
            ),
          };
          shader.uniforms.grassDraw = {
            value: new THREE.Vector4(GRASS_LOD_DIST, GRASS_MAX_DIST, 0, 0),
          };
          shader.uniforms.time = { value: 0.0 };
          shader.uniforms.heightmap = { value: hmap };
          shader.uniforms.heightParams = {
            value: new THREE.Vector4(100, 0, 0, 0), // terrainSize - will be updated via useEffect
          };
          shader.uniforms.playerPos = { value: new THREE.Vector3(0, 0, 0) };
          shader.uniforms.viewMatrixInverse = { value: new THREE.Matrix4() };

          // Wind uniforms
          shader.uniforms.windParams = {
            value: new THREE.Vector4(
              windDirectionScale,
              windDirectionSpeed,
              windStrengthScale,
              windStrengthSpeed
            ),
          };
          shader.uniforms.windStrength = {
            value: windEnabled ? windStrength : 0.0,
          };

          // Player interaction uniforms
          shader.uniforms.playerInteractionParams = {
            value: new THREE.Vector2(
              playerInteractionEnabled ? playerInteractionRange : 999.0,
              playerInteractionEnabled ? playerInteractionStrength : 0.0
            ),
          };
          shader.uniforms.playerInteractionRepel = {
            value: playerInteractionRepel ? 1.0 : -1.0,
          };
          shader.uniforms.playerInteractionHeightThreshold = {
            value: playerInteractionHeightThreshold,
          };

          // Grass color uniforms
          shader.uniforms.uBaseColor1 = {
            value: baseColor1Ref.current.clone(),
          };
          shader.uniforms.uBaseColor2 = {
            value: baseColor2Ref.current.clone(),
          };
          shader.uniforms.uTipColor1 = { value: tipColor1Ref.current.clone() };
          shader.uniforms.uTipColor2 = { value: tipColor2Ref.current.clone() };
          shader.uniforms.uGradientCurve = { value: gradientCurve };
          shader.uniforms.uAoEnabled = { value: aoEnabled };
          shader.uniforms.uAoIntensity = { value: aoIntensity };
          shader.uniforms.uFogEnabled = { value: fogEnabled };
          shader.uniforms.uFogNear = { value: fogNear };
          shader.uniforms.uFogFar = { value: fogFar };
          shader.uniforms.uFogIntensity = { value: fogIntensity };
          shader.uniforms.uFogColor = { value: fogColorRef.current.clone() };
          shader.uniforms.uSpecularEnabled = { value: specularEnabled };
          shader.uniforms.uSpecularIntensity = { value: specularIntensity };
          shader.uniforms.uSpecularColor = {
            value: specularColorRef.current.clone(),
          };
          shader.uniforms.uSpecularDirection = {
            value: new THREE.Vector3(
              specularDirectionX,
              specularDirectionY,
              specularDirectionZ
            ).normalize(),
          };
          shader.uniforms.uGrassMiddleBrightnessMin = {
            value: grassMiddleBrightnessMin,
          };
          shader.uniforms.uGrassMiddleBrightnessMax = {
            value: grassMiddleBrightnessMax,
          };
          shader.uniforms.uBackscatterEnabled = { value: backscatterEnabled };
          shader.uniforms.uBackscatterIntensity = {
            value: backscatterIntensity,
          };
          shader.uniforms.uBackscatterColor = {
            value: backscatterColorRef.current.clone(),
          };
          shader.uniforms.uBackscatterPower = { value: backscatterPower };
          shader.uniforms.uFrontScatterStrength = {
            value: frontScatterStrength,
          };
          shader.uniforms.uRimSSSStrength = { value: rimSSSStrength };

          // Replace shaders with complete versions
          shader.vertexShader = vertexShader;
          shader.fragmentShader = fragmentShader;

          // Store reference for updates
          mat.userData.shader = shader;
        };

        mat.needsUpdate = true;

        return mat;
      };

      const matLow = createMaterial(GRASS_SEGMENTS_LOW, GRASS_VERTICES_LOW);
      const matHigh = createMaterial(GRASS_SEGMENTS_HIGH, GRASS_VERTICES_HIGH);

      return {
        geometryLow: geoLow,
        geometryHigh: geoHigh,
        materialLow: matLow,
        materialHigh: matHigh,
        heightmap: hmap,
      };
    }, [grassDensity]); // Recreate when grassDensity changes

  // Mesh pools
  const meshPoolLow = useRef([]);
  const meshPoolHigh = useRef([]);
  const totalTime = useRef(0);

  // Track player position - handle both Vector3 and array formats
  const playerPosRef = useRef(new THREE.Vector3(0, 0, 0));

  // Update player position ref when prop changes
  useEffect(() => {
    if (playerPosition) {
      if (Array.isArray(playerPosition)) {
        playerPosRef.current.set(
          playerPosition[0] || 0,
          playerPosition[1] || 0,
          playerPosition[2] || 0
        );
      } else if (playerPosition instanceof THREE.Vector3) {
        playerPosRef.current.copy(playerPosition);
      } else if (playerPosition.x !== undefined) {
        // Handle object with x, y, z properties
        playerPosRef.current.set(
          playerPosition.x || 0,
          playerPosition.y || 0,
          playerPosition.z || 0
        );
      }
    }
  }, [playerPosition]);

  // Update function
  useFrame((state, delta) => {
    if (!groupRef.current) return;

    totalTime.current += delta;

    // Update player position from prop (in case it's a reactive ref/object)
    if (playerPosition) {
      if (Array.isArray(playerPosition)) {
        playerPosRef.current.set(
          playerPosition[0] || 0,
          playerPosition[1] || 0,
          playerPosition[2] || 0
        );
      } else if (playerPosition instanceof THREE.Vector3) {
        playerPosRef.current.copy(playerPosition);
      } else if (playerPosition.x !== undefined) {
        playerPosRef.current.set(
          playerPosition.x || 0,
          playerPosition.y || 0,
          playerPosition.z || 0
        );
      }
    }

    // Debug player position every 60 frames
    if (Math.floor(totalTime.current * 60) % 60 === 0) {
      console.log("🌿 ClaudeGrassQuick5 playerPosition prop:", playerPosition);
      console.log("🌿 ClaudeGrassQuick5 playerPosRef:", playerPosRef.current);
    }

    // Update shader uniforms
    if (materialLow.userData.shader) {
      materialLow.userData.shader.uniforms.time.value = totalTime.current;
      materialLow.userData.shader.uniforms.playerPos.value.copy(
        playerPosRef.current
      );
      materialLow.userData.shader.uniforms.viewMatrixInverse.value =
        camera.matrixWorld;
      // Update control-based uniforms
      materialLow.userData.shader.uniforms.grassSize.value.set(
        grassWidth,
        grassHeight
      );
      materialLow.userData.shader.uniforms.grassParams.value.z = heightScale;
      materialLow.userData.shader.uniforms.grassParams.value.w = heightOffset;
      materialLow.userData.shader.uniforms.grassDraw.value.set(
        lodDistance,
        maxDistance,
        0,
        0
      );
      materialLow.userData.shader.uniforms.heightParams.value.x = terrainSize;
      // Update wind uniforms
      materialLow.userData.shader.uniforms.windParams.value.set(
        windDirectionScale,
        windDirectionSpeed,
        windStrengthScale,
        windStrengthSpeed
      );
      materialLow.userData.shader.uniforms.windStrength.value = windEnabled
        ? windStrength
        : 0.0;
      // Update player interaction uniforms
      materialLow.userData.shader.uniforms.playerInteractionParams.value.set(
        playerInteractionEnabled ? playerInteractionRange : 999.0,
        playerInteractionEnabled ? playerInteractionStrength : 0.0
      );
      materialLow.userData.shader.uniforms.playerInteractionRepel.value =
        playerInteractionRepel ? 1.0 : -1.0;
      materialLow.userData.shader.uniforms.playerInteractionHeightThreshold.value =
        playerInteractionHeightThreshold;

      // Update grass color uniforms
      materialLow.userData.shader.uniforms.uBaseColor1.value.copy(
        baseColor1Ref.current
      );
      materialLow.userData.shader.uniforms.uBaseColor2.value.copy(
        baseColor2Ref.current
      );
      materialLow.userData.shader.uniforms.uTipColor1.value.copy(
        tipColor1Ref.current
      );
      materialLow.userData.shader.uniforms.uTipColor2.value.copy(
        tipColor2Ref.current
      );
      materialLow.userData.shader.uniforms.uGradientCurve.value = gradientCurve;
      materialLow.userData.shader.uniforms.uAoEnabled.value = aoEnabled;
      materialLow.userData.shader.uniforms.uAoIntensity.value = aoIntensity;
      materialLow.userData.shader.uniforms.uFogEnabled.value = fogEnabled;
      materialLow.userData.shader.uniforms.uFogNear.value = fogNear;
      materialLow.userData.shader.uniforms.uFogFar.value = fogFar;
      materialLow.userData.shader.uniforms.uFogIntensity.value = fogIntensity;
      materialLow.userData.shader.uniforms.uFogColor.value.copy(
        fogColorRef.current
      );
      materialLow.userData.shader.uniforms.uSpecularEnabled.value =
        specularEnabled;
      materialLow.userData.shader.uniforms.uSpecularIntensity.value =
        specularIntensity;
      materialLow.userData.shader.uniforms.uSpecularColor.value.copy(
        specularColorRef.current
      );
      materialLow.userData.shader.uniforms.uSpecularDirection.value
        .set(specularDirectionX, specularDirectionY, specularDirectionZ)
        .normalize();
      materialLow.userData.shader.uniforms.uGrassMiddleBrightnessMin.value =
        grassMiddleBrightnessMin;
      materialLow.userData.shader.uniforms.uGrassMiddleBrightnessMax.value =
        grassMiddleBrightnessMax;
      materialLow.userData.shader.uniforms.uBackscatterEnabled.value =
        backscatterEnabled;
      materialLow.userData.shader.uniforms.uBackscatterIntensity.value =
        backscatterIntensity;
      materialLow.userData.shader.uniforms.uBackscatterColor.value.copy(
        backscatterColorRef.current
      );
      materialLow.userData.shader.uniforms.uBackscatterPower.value =
        backscatterPower;
      materialLow.userData.shader.uniforms.uFrontScatterStrength.value =
        frontScatterStrength;
      materialLow.userData.shader.uniforms.uRimSSSStrength.value =
        rimSSSStrength;

      // Debug log uniforms every 60 frames
      if (Math.floor(totalTime.current * 60) % 60 === 0) {
        console.log("🌿 Shader uniforms:", {
          playerPos: materialLow.userData.shader.uniforms.playerPos.value,
          playerInteractionParams:
            materialLow.userData.shader.uniforms.playerInteractionParams.value,
          playerInteractionEnabled,
          playerInteractionRange,
          playerInteractionStrength,
        });
      }
    }
    if (materialHigh.userData.shader) {
      materialHigh.userData.shader.uniforms.time.value = totalTime.current;
      materialHigh.userData.shader.uniforms.playerPos.value.copy(
        playerPosRef.current
      );
      materialHigh.userData.shader.uniforms.viewMatrixInverse.value =
        camera.matrixWorld;
      // Update control-based uniforms
      materialHigh.userData.shader.uniforms.grassSize.value.set(
        grassWidth,
        grassHeight
      );
      materialHigh.userData.shader.uniforms.grassParams.value.z = heightScale;
      materialHigh.userData.shader.uniforms.grassParams.value.w = heightOffset;
      materialHigh.userData.shader.uniforms.grassDraw.value.set(
        lodDistance,
        maxDistance,
        0,
        0
      );
      materialHigh.userData.shader.uniforms.heightParams.value.x = terrainSize;
      // Update wind uniforms
      materialHigh.userData.shader.uniforms.windParams.value.set(
        windDirectionScale,
        windDirectionSpeed,
        windStrengthScale,
        windStrengthSpeed
      );
      materialHigh.userData.shader.uniforms.windStrength.value = windEnabled
        ? windStrength
        : 0.0;
      // Update player interaction uniforms
      materialHigh.userData.shader.uniforms.playerInteractionParams.value.set(
        playerInteractionEnabled ? playerInteractionRange : 999.0,
        playerInteractionEnabled ? playerInteractionStrength : 0.0
      );
      materialHigh.userData.shader.uniforms.playerInteractionRepel.value =
        playerInteractionRepel ? 1.0 : -1.0;
      materialHigh.userData.shader.uniforms.playerInteractionHeightThreshold.value =
        playerInteractionHeightThreshold;

      // Update grass color uniforms
      materialHigh.userData.shader.uniforms.uBaseColor1.value.copy(
        baseColor1Ref.current
      );
      materialHigh.userData.shader.uniforms.uBaseColor2.value.copy(
        baseColor2Ref.current
      );
      materialHigh.userData.shader.uniforms.uTipColor1.value.copy(
        tipColor1Ref.current
      );
      materialHigh.userData.shader.uniforms.uTipColor2.value.copy(
        tipColor2Ref.current
      );
      materialHigh.userData.shader.uniforms.uGradientCurve.value =
        gradientCurve;
      materialHigh.userData.shader.uniforms.uAoEnabled.value = aoEnabled;
      materialHigh.userData.shader.uniforms.uAoIntensity.value = aoIntensity;
      materialHigh.userData.shader.uniforms.uFogEnabled.value = fogEnabled;
      materialHigh.userData.shader.uniforms.uFogNear.value = fogNear;
      materialHigh.userData.shader.uniforms.uFogFar.value = fogFar;
      materialHigh.userData.shader.uniforms.uFogIntensity.value = fogIntensity;
      materialHigh.userData.shader.uniforms.uFogColor.value.copy(
        fogColorRef.current
      );
      materialHigh.userData.shader.uniforms.uSpecularEnabled.value =
        specularEnabled;
      materialHigh.userData.shader.uniforms.uSpecularIntensity.value =
        specularIntensity;
      materialHigh.userData.shader.uniforms.uSpecularColor.value.copy(
        specularColorRef.current
      );
      materialHigh.userData.shader.uniforms.uSpecularDirection.value
        .set(specularDirectionX, specularDirectionY, specularDirectionZ)
        .normalize();
      materialHigh.userData.shader.uniforms.uGrassMiddleBrightnessMin.value =
        grassMiddleBrightnessMin;
      materialHigh.userData.shader.uniforms.uGrassMiddleBrightnessMax.value =
        grassMiddleBrightnessMax;
      materialHigh.userData.shader.uniforms.uBackscatterEnabled.value =
        backscatterEnabled;
      materialHigh.userData.shader.uniforms.uBackscatterIntensity.value =
        backscatterIntensity;
      materialHigh.userData.shader.uniforms.uBackscatterColor.value.copy(
        backscatterColorRef.current
      );
      materialHigh.userData.shader.uniforms.uBackscatterPower.value =
        backscatterPower;
      materialHigh.userData.shader.uniforms.uFrontScatterStrength.value =
        frontScatterStrength;
      materialHigh.userData.shader.uniforms.uRimSSSStrength.value =
        rimSSSStrength;
    }

    // Frustum culling setup
    const frustum = new THREE.Frustum();
    const projScreenMatrix = new THREE.Matrix4();
    projScreenMatrix.multiplyMatrices(
      camera.projectionMatrix,
      camera.matrixWorldInverse
    );
    frustum.setFromProjectionMatrix(projScreenMatrix);

    // Calculate base cell position using patchSpacing
    const baseCellPos = camera.position.clone();
    baseCellPos.divideScalar(patchSpacing);
    baseCellPos.floor();
    baseCellPos.multiplyScalar(patchSpacing);

    // Hide all meshes
    groupRef.current.children.forEach((child) => {
      child.visible = false;
    });

    const meshesLowAvailable = [...meshPoolLow.current];
    const meshesHighAvailable = [...meshPoolHigh.current];

    const cameraPosXZ = new THREE.Vector3(
      camera.position.x,
      0,
      camera.position.z
    );
    const aabbTmp = new THREE.Box3();

    // Grid of patches around camera using gridSize
    for (let x = -gridSize; x < gridSize; x++) {
      for (let z = -gridSize; z < gridSize; z++) {
        const currentCell = new THREE.Vector3(
          baseCellPos.x + x * patchSpacing,
          0,
          baseCellPos.z + z * patchSpacing
        );

        aabbTmp.setFromCenterAndSize(
          currentCell,
          new THREE.Vector3(patchSpacing, 1000, patchSpacing)
        );

        const distToCell = aabbTmp.distanceToPoint(cameraPosXZ);

        // Distance culling
        if (distToCell > GRASS_MAX_DIST) continue;

        // Frustum culling
        if (!frustum.intersectsBox(aabbTmp)) continue;

        // LOD selection
        const useLowLOD = distToCell > GRASS_LOD_DIST;
        const meshPool = useLowLOD ? meshesLowAvailable : meshesHighAvailable;
        const geo = useLowLOD ? geometryLow : geometryHigh;
        const mat = useLowLOD ? materialLow : materialHigh;

        let mesh;
        if (meshPool.length > 0) {
          mesh = meshPool.pop();
        } else {
          // Create new mesh if pool is empty
          mesh = new THREE.Mesh(geo, mat);
          mesh.receiveShadow = true;
          mesh.castShadow = false;
          groupRef.current.add(mesh);

          if (useLowLOD) {
            meshPoolLow.current.push(mesh);
          } else {
            meshPoolHigh.current.push(mesh);
          }
        }

        mesh.position.copy(currentCell);
        mesh.position.y = 0;
        mesh.visible = true;
      }
    }
  });

  return <group ref={groupRef} />;
}
