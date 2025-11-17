import * as THREE from 'three';
import { useRef, useMemo, useEffect } from 'react';
import { useFrame } from '@react-three/fiber';
import GFS3_Simple from './GFS3_Simple';

// Constants from Simon Dev's implementation
const NUM_GRASS_X = 32;
const NUM_GRASS_Y = 32;
const NUM_GRASS = (NUM_GRASS_X * NUM_GRASS_Y) * 3;

const GRASS_SEGMENTS_LOW = 1;
const GRASS_SEGMENTS_HIGH = 6;
const GRASS_VERTICES_LOW = (GRASS_SEGMENTS_LOW + 1) * 2;
const GRASS_VERTICES_HIGH = (GRASS_SEGMENTS_HIGH + 1) * 2;

const GRASS_LOD_DIST = 15;
const GRASS_MAX_DIST = 100;
const GRASS_PATCH_SIZE = 5 * 2;
const GRASS_WIDTH = 0.1;
const GRASS_HEIGHT = 1.5;

// Custom InstancedFloat16BufferAttribute
class InstancedFloat16BufferAttribute extends THREE.InstancedBufferAttribute {
  constructor(array, itemSize, normalized, meshPerAttribute = 1) {
    super(new Uint16Array(array), itemSize, normalized, meshPerAttribute);
    this.isFloat16BufferAttribute = true;
  }
}

// Simple random number generator
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

function CreateIndexBuffer(segments) {
  const VERTICES = (segments + 1) * 2;
  const indices = [];

  for (let i = 0; i < segments; ++i) {
    const vi = i * 2;
    indices[i*12+0] = vi + 0;
    indices[i*12+1] = vi + 1;
    indices[i*12+2] = vi + 2;

    indices[i*12+3] = vi + 2;
    indices[i*12+4] = vi + 1;
    indices[i*12+5] = vi + 3;

    const fi = VERTICES + vi;
    indices[i*12+6] = fi + 2;
    indices[i*12+7] = fi + 1;
    indices[i*12+8] = fi + 0;

    indices[i*12+9]  = fi + 3;
    indices[i*12+10] = fi + 1;
    indices[i*12+11] = fi + 2;
  }

  return indices;
}

function CreateTileGeometry(segments) {
  setSeed(0);

  const VERTICES = (segments + 1) * 2;

  const indices = CreateIndexBuffer(segments);

  const offsets = [];
  for (let i = 0; i < NUM_GRASS; ++i) {
    offsets.push(randRange(-GRASS_PATCH_SIZE * 0.5, GRASS_PATCH_SIZE * 0.5));
    offsets.push(randRange(-GRASS_PATCH_SIZE * 0.5, GRASS_PATCH_SIZE * 0.5));
    offsets.push(0);
  }

  const offsetsData = offsets.map(THREE.DataUtils.toHalfFloat);

  const vertID = new Uint8Array(VERTICES*2);
  for (let i = 0; i < VERTICES*2; ++i) {
    vertID[i] = i;
  }

  // Create base geometry positions (one per vertex, not instanced)
  // These will be overridden by the shader, but Three.js needs them
  const basePositions = new Float32Array(VERTICES * 2 * 3); // * 3 for x,y,z
  for (let i = 0; i < basePositions.length; i++) {
    basePositions[i] = 0; // All zeros, shader will calculate actual positions
  }

  const geo = new THREE.InstancedBufferGeometry();
  geo.instanceCount = NUM_GRASS;
  geo.setAttribute('position', new THREE.Float32BufferAttribute(basePositions, 3)); // Base geometry
  geo.setAttribute('vertIndex', new THREE.Uint8BufferAttribute(vertID, 1));
  geo.setAttribute('offset', new InstancedFloat16BufferAttribute(offsetsData, 3)); // Instanced offsets
  geo.setIndex(indices);
  geo.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 0, 0), 1 + GRASS_PATCH_SIZE * 2);

  return geo;
}

// Shader code - combining all the glsl files Simon Dev uses
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

// Custom varyings and uniforms
varying vec3 vGrassColour;
varying vec4 vGrassParams;
varying vec3 vNormal2;
varying vec3 vGrassWorldPosition;

uniform vec2 grassSize;
uniform vec4 grassParams;
uniform vec4 grassDraw;
uniform float time;
uniform sampler2D heightmap;
uniform vec4 heightParams;
uniform vec3 playerPos;
uniform mat4 viewMatrixInverse;

attribute float vertIndex;
attribute vec3 offset;

// Utility functions (saturate is already defined in Three.js common)
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

// Hash functions for noise
uint murmurHash11(uint src) {
  const uint M = 0x5bd1e995u;
  uint h = 1190494759u;
  src *= M; src ^= src>>24u; src *= M;
  h *= M; h ^= src;
  h ^= h>>13u; h *= M; h ^= h>>15u;
  return h;
}

uint murmurHash12(uvec2 src) {
  const uint M = 0x5bd1e995u;
  uint h = 1190494759u;
  src *= M; src ^= src>>24u; src *= M;
  h *= M; h ^= src.x; h *= M; h ^= src.y;
  h ^= h>>13u; h *= M; h ^= h>>15u;
  return h;
}

uvec2 murmurHash22(uvec2 src) {
  const uint M = 0x5bd1e995u;
  uvec2 h = uvec2(1190494759u, 2147483647u);
  src *= M; src ^= src>>24u; src *= M;
  h *= M; h ^= src.x; h *= M; h ^= src.y;
  h ^= h>>13u; h *= M; h ^= h>>15u;
  return h;
}

uvec4 murmurHash42(uvec2 src) {
    const uint M = 0x5bd1e995u;
    uvec4 h = uvec4(1190494759u, 2147483647u, 3559788179u, 179424673u);
    src *= M; src ^= src>>24u; src *= M;
    h *= M; h ^= src.x; h *= M; h ^= src.y;
    h ^= h>>13u; h *= M; h ^= h>>15u;
    return h;
}

float hash12(vec2 src) {
  uint h = murmurHash12(floatBitsToUint(src));
  return uintBitsToFloat(h & 0x007fffffu | 0x3f800000u) - 1.0;
}

vec2 hash22(vec2 src) {
  uvec2 h = murmurHash22(floatBitsToUint(src));
  return uintBitsToFloat(h & 0x007fffffu | 0x3f800000u) - 1.0;
}

vec4 hash42(vec2 src) {
  uvec4 h = murmurHash42(floatBitsToUint(src));
  return uintBitsToFloat(h & 0x007fffffu | 0x3f800000u) - 1.0;
}

float noise12(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  vec2 u = smoothstep(vec2(0.0), vec2(1.0), f);

  float val = mix( mix( hash12( i + vec2(0.0, 0.0) ),
                        hash12( i + vec2(1.0, 0.0) ), u.x),
                   mix( hash12( i + vec2(0.0, 1.0) ),
                        hash12( i + vec2(1.0, 1.0) ), u.x), u.y);
  return val * 2.0 - 1.0;
}

void main() {
  #include <uv_vertex>
  #include <color_vertex>
  #include <morphcolor_vertex>
  #include <beginnormal_vertex>
  #include <begin_vertex>

  vec3 grassOffset = vec3(offset.x, 0.0, offset.y);

  // Blade world position
  vec3 grassBladeWorldPos = (modelMatrix * vec4(grassOffset, 1.0)).xyz;
  vec2 heightmapUV = vec2(
      remap(grassBladeWorldPos.x, -heightParams.x * 0.5, heightParams.x * 0.5, 0.0, 1.0),
      remap(grassBladeWorldPos.z, -heightParams.x * 0.5, heightParams.x * 0.5, 1.0, 0.0));
  vec4 heightmapSample = texture2D(heightmap, heightmapUV);
  grassBladeWorldPos.y += heightmapSample.x * grassParams.z - grassParams.w;

  float heightmapSampleHeight = 1.0;

  vec4 hashVal1 = hash42(vec2(grassBladeWorldPos.x, grassBladeWorldPos.z));

  float highLODOut = smoothstep(grassDraw.x * 0.5, grassDraw.x, distance(cameraPosition, grassBladeWorldPos));
  float lodFadeIn = smoothstep(grassDraw.x, grassDraw.y, distance(cameraPosition, grassBladeWorldPos));

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

  // 1 = front, -1 = back
  float zSide = -(floor(vertIndex / GRASS_VERTICES) * 2.0 - 1.0);

  // 0 = left, 1 = right
  float xSide = mod(vertID, 2.0);

  float heightPercent = (vertID - xSide) / (GRASS_SEGMENTS * 2.0);

  float grassTotalHeight = grassSize.y * randomHeight;
  float grassTotalWidthHigh = easeOut(1.0 - heightPercent, 2.0);
  float grassTotalWidthLow = 1.0 - heightPercent;
  float grassTotalWidth = grassSize.x * mix(grassTotalWidthHigh, grassTotalWidthLow, highLODOut) * randomWidth;

  // Shift verts
  float x = (xSide - 0.5) * grassTotalWidth;
  float y = heightPercent * grassTotalHeight;

  float windDir = noise12(grassBladeWorldPos.xz * 0.05 + 0.05 * time);
  float windNoiseSample = noise12(grassBladeWorldPos.xz * 0.25 + time * 1.0);
  float windLeanAngle = remap(windNoiseSample, -1.0, 1.0, 0.25, 1.0);
  windLeanAngle = easeIn(windLeanAngle, 2.0) * 1.25;
  vec3 windAxis = vec3(cos(windDir), 0.0, sin(windDir));

  windLeanAngle *= heightPercent;

  float distToPlayer = distance(grassBladeWorldPos.xz, playerPos.xz);
  float playerFalloff = smoothstep(2.5, 1.0, distToPlayer);
  float playerLeanAngle = mix(0.0, 0.2, playerFalloff * linearstep(0.5, 0.0, windLeanAngle));
  vec3 grassToPlayer = normalize(vec3(playerPos.x, 0.0, playerPos.z) - vec3(grassBladeWorldPos.x, 0.0, grassBladeWorldPos.z));
  vec3 playerLeanAxis = vec3(grassToPlayer.z, 0, -grassToPlayer.x);

  randomLean += leanAnimation;

  float easedHeight = mix(easeIn(heightPercent, 2.0), 1.0, highLODOut);
  float curveAmount = -randomLean * easedHeight;

  float ncurve1 = -randomLean * easedHeight;
  vec3 n1 = vec3(0.0, (heightPercent + 0.01), 0.0);
  n1 = rotateX(ncurve1) * n1;

  float ncurve2 = -randomLean * easedHeight * 0.9;
  vec3 n2 = vec3(0.0, (heightPercent + 0.01) * 0.9, 0.0);
  n2 = rotateX(ncurve2) * n2;

  vec3 ncurve = normalize(n1 - n2);

  mat3 grassMat = rotateAxis(playerLeanAxis, playerLeanAngle) * rotateAxis(windAxis, windLeanAngle) * rotateY(randomAngle);

  vec3 grassFaceNormal = vec3(0.0, 0.0, 1.0);
  grassFaceNormal = grassMat * grassFaceNormal;
  grassFaceNormal *= zSide;

  vec3 grassVertexNormal = vec3(0.0, -ncurve.z, ncurve.y);
  vec3 grassVertexNormal1 = rotateY(PI * 0.3 * zSide) * grassVertexNormal;
  vec3 grassVertexNormal2 = rotateY(PI * -0.3 * zSide) * grassVertexNormal;

  grassVertexNormal1 = grassMat * grassVertexNormal1;
  grassVertexNormal1 *= zSide;

  grassVertexNormal2 = grassMat * grassVertexNormal2;
  grassVertexNormal2 *= zSide;

  vec3 grassVertexPosition = vec3(x, y, 0.0);
  grassVertexPosition = rotateX(curveAmount) * grassVertexPosition;
  grassVertexPosition = grassMat * grassVertexPosition;

  grassVertexPosition += grassOffset;

  vec3 b1 = vec3(0.02, 0.075, 0.01);
  vec3 b2 = vec3(0.025, 0.1, 0.01);
  vec3 t1 = vec3(0.65, 0.8, 0.25);
  vec3 t2 = vec3(0.8, 0.9, 0.4);

  vec3 baseColour = mix(b1, b2, hashGrassColour.x);
  vec3 tipColour = mix(t1, t2, hashGrassColour.y);
  vec3 highLODColour = mix(baseColour, tipColour, easeIn(heightPercent, 4.0)) * randomShade;
  vec3 lowLODColour = mix(b1, t1, heightPercent);
  vGrassColour = mix(highLODColour, lowLODColour, highLODOut);
  vGrassParams = vec4(heightPercent, grassBladeWorldPos.y, highLODOut, xSide);

  const float SKY_RATIO = 0.25;
  vec3 UP = vec3(0.0, 1.0, 0.0);
  float skyFadeIn = (1.0 - highLODOut) * SKY_RATIO;
  vec3 normal1 = normalize(mix(UP, grassVertexNormal1, skyFadeIn));
  vec3 normal2 = normalize(mix(UP, grassVertexNormal2, skyFadeIn));

  transformed = grassVertexPosition;
  transformed.y += grassBladeWorldPos.y;

  vec3 viewDir = normalize(cameraPosition - grassBladeWorldPos);
  vec3 viewDirXZ = normalize(vec3(viewDir.x, 0.0, viewDir.z));

  vec3 grassFaceNormalXZ = normalize(vec3(grassFaceNormal.x, 0.0, grassFaceNormal.z));

  float viewDotNormal = saturate(dot(grassFaceNormal, viewDirXZ));
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

  vec4 mvPosition = vec4( transformed, 1.0 );
  #ifdef USE_INSTANCING
    mvPosition = instanceMatrix * mvPosition;
  #endif
  mvPosition = modelViewMatrix * mvPosition;

  mvPosition.x += viewSpaceThickenFactor * (xSide - 0.5) * grassTotalWidth * 0.5 * zSide;

  gl_Position = projectionMatrix * mvPosition;

  #include <logdepthbuf_vertex>
  #include <clipping_planes_vertex>
  vViewPosition = - mvPosition.xyz;
  #include <worldpos_vertex>
  #include <envmap_vertex>
  #include <shadowmap_vertex>
  #include <fog_vertex>

  vGrassWorldPosition = worldPosition.xyz;
}
`;

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

uniform vec3 grassLODColour;
uniform float time;

varying vec3 vGrassColour;
varying vec4 vGrassParams;
varying vec3 vNormal2;
varying vec3 vGrassWorldPosition;

// Utility functions (saturate is already defined in Three.js common)
float linearstep(float minValue, float maxValue, float v) {
  return clamp((v - minValue) / (maxValue - minValue), 0.0, 1.0);
}

float easeIn(float x, float t) {
  return pow(x, t);
}

#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <specularmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>

void main() {
  vec3 viewDir = normalize(cameraPosition - vGrassWorldPosition);

  #include <clipping_planes_fragment>
  vec4 diffuseColor = vec4( diffuse, opacity );

  float heightPercent = vGrassParams.x;
  float height = vGrassParams.y;
  float lodFadeIn = vGrassParams.z;
  float lodFadeOut = 1.0 - lodFadeIn;

  float grassMiddle = mix(
      smoothstep(abs(vGrassParams.w - 0.5), 0.0, 0.1), 1.0, lodFadeIn);

  float isSandy = saturate(linearstep(-11.0, -14.0, height));
  float density = 1.0 - isSandy;

  float aoForDensity = mix(1.0, 0.25, density);
  float ao = mix(aoForDensity, 1.0, easeIn(heightPercent, 2.0));

  diffuseColor.rgb *= vGrassColour;
  diffuseColor.rgb *= mix(0.85, 1.0, grassMiddle);
  diffuseColor.rgb *= ao;

  ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
  vec3 totalEmissiveRadiance = emissive;

  #include <logdepthbuf_fragment>
  #include <map_fragment>
  #include <color_fragment>
  #include <alphamap_fragment>
  #include <alphatest_fragment>
  #include <alphahash_fragment>
  #include <specularmap_fragment>
  #include <normal_fragment_begin>
  #include <normal_fragment_maps>

  vec3 normal2 = normalize(vNormal2);
  normal = normalize(mix(vNormal, normal2, vGrassParams.w));

  #include <emissivemap_fragment>

  // Phong lighting
  #include <lights_phong_fragment>
  #include <lights_fragment_begin>
  #include <lights_fragment_maps>
  #include <lights_fragment_end>
  #include <aomap_fragment>

  vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + reflectedLight.directSpecular + reflectedLight.indirectSpecular + totalEmissiveRadiance;

  #include <envmap_fragment>
  #include <opaque_fragment>
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
  #include <fog_fragment>
  #include <premultiplied_alpha_fragment>
  #include <dithering_fragment>
}
`;

export default function GFS3({
  heightData,
  terrainSize,
  terrainSegments,
  terrainOffset,
  terrainHeight
}) {
  const groupRef = useRef();
  const timeRef = useRef(0);
  const materialsRef = useRef({ low: null, high: null });

  // Create geometries
  const { geometryLow, geometryHigh } = useMemo(() => {
    return {
      geometryLow: CreateTileGeometry(GRASS_SEGMENTS_LOW),
      geometryHigh: CreateTileGeometry(GRASS_SEGMENTS_HIGH)
    };
  }, []);

  // Create heightmap texture from heightData
  const heightmapTexture = useMemo(() => {
    if (!heightData) return null;

    const heightmapResolution = 512;
    const textureData = new Float32Array(heightmapResolution * heightmapResolution);

    for (let y = 0; y < heightmapResolution; y++) {
      for (let x = 0; x < heightmapResolution; x++) {
        const u = x / (heightmapResolution - 1);
        const v = y / (heightmapResolution - 1);

        const dataX = u * terrainSegments;
        const dataY = v * terrainSegments;

        const x0 = Math.floor(dataX);
        const y0 = Math.floor(dataY);
        const x1 = Math.min(x0 + 1, terrainSegments);
        const y1 = Math.min(y0 + 1, terrainSegments);

        const fx = dataX - x0;
        const fy = dataY - y0;

        const h00 = heightData[y0 * (terrainSegments + 1) + x0] / terrainHeight;
        const h10 = heightData[y0 * (terrainSegments + 1) + x1] / terrainHeight;
        const h01 = heightData[y1 * (terrainSegments + 1) + x0] / terrainHeight;
        const h11 = heightData[y1 * (terrainSegments + 1) + x1] / terrainHeight;

        const h0 = h00 * (1 - fx) + h10 * fx;
        const h1 = h01 * (1 - fx) + h11 * fx;
        const height = h0 * (1 - fy) + h1 * fy;

        textureData[y * heightmapResolution + x] = Math.max(0, Math.min(1, height));
      }
    }

    const texture = new THREE.DataTexture(
      textureData,
      heightmapResolution,
      heightmapResolution,
      THREE.RedFormat,
      THREE.FloatType
    );
    texture.needsUpdate = true;
    texture.wrapS = THREE.ClampToEdgeWrapping;
    texture.wrapT = THREE.ClampToEdgeWrapping;
    texture.minFilter = THREE.LinearFilter;
    texture.magFilter = THREE.LinearFilter;

    return texture;
  }, [heightData, terrainSegments, terrainHeight]);

  // Create materials
  const createMaterial = (segments, vertices) => {
    const material = new THREE.MeshPhongMaterial({
      side: THREE.FrontSide,
      alphaTest: 0.5,
    });

    material.onBeforeCompile = (shader) => {
      shader.vertexShader = grassVertexShader;
      shader.fragmentShader = grassFragmentShader;

      shader.uniforms.time = { value: 0.0 };
      shader.uniforms.playerPos = { value: new THREE.Vector3(0, 0, 0) };
      shader.uniforms.grassSize = { value: new THREE.Vector2(GRASS_WIDTH, GRASS_HEIGHT) };
      shader.uniforms.grassParams = { value: new THREE.Vector4(segments, vertices, terrainHeight, Math.abs(terrainOffset)) };
      shader.uniforms.grassDraw = { value: new THREE.Vector4(GRASS_LOD_DIST, GRASS_MAX_DIST, 0, 0) };
      shader.uniforms.heightmap = { value: heightmapTexture };
      shader.uniforms.heightParams = { value: new THREE.Vector4(terrainSize, 0, 0, 0) };
      shader.uniforms.grassLODColour = { value: new THREE.Vector3(0, 0, 1) };
      shader.uniforms.viewMatrixInverse = { value: new THREE.Matrix4() };

      material.userData.shader = shader;
    };

    return material;
  };

  const materialLow = useMemo(() => createMaterial(GRASS_SEGMENTS_LOW, GRASS_VERTICES_LOW), [heightmapTexture]);
  const materialHigh = useMemo(() => createMaterial(GRASS_SEGMENTS_HIGH, GRASS_VERTICES_HIGH), [heightmapTexture]);

  useEffect(() => {
    materialsRef.current = { low: materialLow, high: materialHigh };
  }, [materialLow, materialHigh]);

  // Update uniforms each frame
  useFrame((state, delta) => {
    timeRef.current += delta;

    [materialLow, materialHigh].forEach(mat => {
      if (mat.userData.shader) {
        mat.userData.shader.uniforms.time.value = timeRef.current;
        mat.userData.shader.uniforms.playerPos.value.set(0, 0, 0); // TODO: Get actual player position
        mat.userData.shader.uniforms.viewMatrixInverse.value.copy(state.camera.matrixWorld);
      }
    });
  });

  if (!heightmapTexture) {
    console.log('GFS3: heightmapTexture not ready');
    return null;
  }

  console.log('GFS3: Rendering grass');
  console.log('GFS3: Geometry:', geometryHigh);
  console.log('GFS3: Geometry bounds:', geometryHigh.boundingSphere);

  // Create simple test geometry - just one triangle
  const testGeo = new THREE.BufferGeometry();
  const testVerts = new Float32Array([
    0, 0, 0,
    0.5, 0, 0,
    0.25, 1, 0
  ]);
  testGeo.setAttribute('position', new THREE.BufferAttribute(testVerts, 3));

  return (
    <group ref={groupRef}>
      {/* Simple instanced grass test */}
      <GFS3_Simple />

      {/* Debug cube */}
      <mesh position={[0, 2, 0]}>
        <boxGeometry args={[1, 1, 1]} />
        <meshStandardMaterial color="red" />
      </mesh>

      {/* Simple triangle to verify rendering works */}
      <mesh position={[3, 0, 0]} geometry={testGeo}>
        <meshStandardMaterial color="yellow" side={THREE.DoubleSide} />
      </mesh>

      {/* GFS3 grass with custom shader - commented out for now */}
      {/* <mesh
        geometry={geometryHigh}
        material={materialHigh}
        position={[0, 5, 0]}
        receiveShadow
        castShadow={false}
        frustumCulled={false}
      /> */}
    </group>
  );
}
