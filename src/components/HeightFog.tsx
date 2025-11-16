import React, { useEffect, useRef } from "react";
import { useThree } from "@react-three/fiber";
import * as THREE from "three";

interface HeightFogProps {
  enabled?: boolean;
  fogColor?: string;
  fogHeight?: number;
  fogNear?: number;
  fogFar?: number;
}

export const HeightFog: React.FC<HeightFogProps> = ({
  enabled = true,
  fogColor = "#cccccc",
  fogHeight = 50.0,
  fogNear = 1,
  fogFar = 2300,
}) => {
  const { scene } = useThree();
  const shadersModified = useRef(false);
  const originalShaders = useRef<{
    fog_pars_vertex?: string;
    fog_vertex?: string;
    fog_pars_fragment?: string;
    fog_fragment?: string;
  }>({});

  // Modify shader chunks once on mount
  useEffect(() => {
    if (!enabled || shadersModified.current) return;

    // Store originals
    originalShaders.current = {
      fog_pars_vertex: THREE.ShaderChunk.fog_pars_vertex,
      fog_vertex: THREE.ShaderChunk.fog_vertex,
      fog_pars_fragment: THREE.ShaderChunk.fog_pars_fragment,
      fog_fragment: THREE.ShaderChunk.fog_fragment,
    };

    // Modify vertex shader to pass world position
    THREE.ShaderChunk.fog_pars_vertex += `
#ifdef USE_FOG
  varying vec3 vWorldPosition;
#endif
`;

    THREE.ShaderChunk.fog_vertex += `
#ifdef USE_FOG
  vWorldPosition = worldPosition.xyz;
#endif
`;

    // Modify fragment shader to include height uniform
    THREE.ShaderChunk.fog_pars_fragment += `
#ifdef USE_FOG
  varying vec3 vWorldPosition;
  uniform float fogHeight;
#endif
`;

    // Replace fog calculation with height-aware version
    const FOG_APPLIED_LINE =
      "gl_FragColor.rgb = mix( gl_FragColor.rgb, fogColor, fogFactor );";
    THREE.ShaderChunk.fog_fragment = THREE.ShaderChunk.fog_fragment.replace(
      FOG_APPLIED_LINE,
      `
  // Height-based fog factor
  float heightFactor = smoothstep(fogHeight, 0.0, vWorldPosition.y);
  float cameraHeightFactor = smoothstep(fogHeight, 0.0, cameraPosition.y);
  
  // Combine distance fog with height fog
  fogFactor = fogFactor * max(heightFactor, cameraHeightFactor);
  
  ${FOG_APPLIED_LINE}
`
    );

    shadersModified.current = true;

    return () => {
      // Restore original shaders on unmount
      if (originalShaders.current.fog_pars_vertex) {
        THREE.ShaderChunk.fog_pars_vertex =
          originalShaders.current.fog_pars_vertex;
      }
      if (originalShaders.current.fog_vertex) {
        THREE.ShaderChunk.fog_vertex = originalShaders.current.fog_vertex;
      }
      if (originalShaders.current.fog_pars_fragment) {
        THREE.ShaderChunk.fog_pars_fragment =
          originalShaders.current.fog_pars_fragment;
      }
      if (originalShaders.current.fog_fragment) {
        THREE.ShaderChunk.fog_fragment = originalShaders.current.fog_fragment;
      }
      shadersModified.current = false;
    };
  }, [enabled]);

  // Set up scene fog
  useEffect(() => {
    if (enabled) {
      scene.fog = new THREE.Fog(fogColor, fogNear, fogFar);
    } else {
      scene.fog = null;
    }

    return () => {
      scene.fog = null;
    };
  }, [enabled, fogColor, fogNear, fogFar, scene]);

  // Update materials with fogHeight uniform
  useEffect(() => {
    if (!enabled || !shadersModified.current) return;

    let materialsUpdated = 0;

    scene.traverse((object) => {
      if (object instanceof THREE.Mesh) {
        const materials = Array.isArray(object.material)
          ? object.material
          : [object.material];

        materials.forEach((material) => {
          if (
            material instanceof THREE.MeshStandardMaterial ||
            material instanceof THREE.MeshBasicMaterial ||
            material instanceof THREE.MeshLambertMaterial ||
            material instanceof THREE.MeshPhongMaterial ||
            material instanceof THREE.MeshToonMaterial
          ) {
            // Enable fog
            material.fog = true;

            // Chain onBeforeCompile instead of replacing it
            // This preserves existing shader modifications (like grass materials)
            const originalOnBeforeCompile = material.onBeforeCompile;

            // Only set onBeforeCompile if it hasn't been set by us already
            if (!(material as any).userData.heightFogApplied) {
              material.onBeforeCompile = (shader, renderer) => {
                // Call original onBeforeCompile first (preserves grass shader code, etc.)
                if (originalOnBeforeCompile) {
                  originalOnBeforeCompile(shader, renderer);
                }

                // Then add fogHeight uniform
                shader.uniforms.fogHeight = { value: fogHeight };

                // Store the shader for later updates
                (material as any).userData.shader = shader;
              };

              // Mark this material as having fog applied to avoid reapplying
              (material as any).userData.heightFogApplied = true;
            }

            // Force recompilation
            material.needsUpdate = true;
            materialsUpdated++;
          }
        });
      }
    });
  }, [enabled, scene, fogHeight]);

  // Update fogHeight when it changes
  useEffect(() => {
    if (!enabled) return;

    scene.traverse((object) => {
      if (object instanceof THREE.Mesh) {
        const materials = Array.isArray(object.material)
          ? object.material
          : [object.material];

        materials.forEach((material) => {
          const shader = (material as any).userData?.shader;
          if (shader && shader.uniforms.fogHeight) {
            shader.uniforms.fogHeight.value = fogHeight;
          }
        });
      }
    });
  }, [enabled, fogHeight, scene]);

  return null;
};
