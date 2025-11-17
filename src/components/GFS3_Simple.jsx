import * as THREE from 'three';
import { useMemo } from 'react';

// Super simple grass test - just render instanced triangles
export default function GFS3_Simple() {
  const geometry = useMemo(() => {
    // Create a simple triangle blade
    const positions = new Float32Array([
      -0.05, 0, 0,  // bottom left
      0.05, 0, 0,   // bottom right
      0, 0.5, 0,    // top
    ]);

    const geo = new THREE.InstancedBufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setIndex([0, 1, 2]);

    // Create instance offsets - 10x10 grid
    const offsets = [];
    for (let x = 0; x < 10; x++) {
      for (let z = 0; z < 10; z++) {
        offsets.push(x * 0.5 - 2.5, 0, z * 0.5 - 2.5);
      }
    }

    geo.setAttribute('offset', new THREE.InstancedBufferAttribute(new Float32Array(offsets), 3));
    geo.instanceCount = 100;

    return geo;
  }, []);

  const material = useMemo(() => {
    const mat = new THREE.MeshBasicMaterial({
      color: 0x00ff00,
      side: THREE.DoubleSide
    });

    mat.onBeforeCompile = (shader) => {
      shader.vertexShader = shader.vertexShader.replace(
        '#include <begin_vertex>',
        `
        #include <begin_vertex>
        transformed += offset;
        `
      );

      shader.vertexShader = 'attribute vec3 offset;\n' + shader.vertexShader;
    };

    return mat;
  }, []);

  return (
    <mesh geometry={geometry} material={material} position={[0, 2, 0]} />
  );
}
