import { RigidBody } from "@react-three/rapier";
import { useFrame } from "@react-three/fiber";
import { useRef } from "react";
import * as THREE from "three";
import ClaudeGrassQuick3 from "./ClaudeGrassQuick3";

export const SideScrollerMap = ({
  playerPosition = [0, 0, 0],
  grassControls,
  ...props
}) => {
  const groundRef = useRef();
  const playerPosRef = useRef(new THREE.Vector3(...playerPosition));

  // Debug: Log controls structure (only once)
  if (grassControls && grassControls.enabled) {
    console.log("🌿 Grass Enabled! Controls:", grassControls);
  }

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

  return (
    <group {...props}>
      {/* Ground plane */}
      <RigidBody type="fixed" colliders="trimesh">
        <mesh
          ref={groundRef}
          rotation={[-Math.PI / 2, 0, 0]}
          position={[0, 0, 0]}
          receiveShadow
        >
          <planeGeometry args={[200, 50]} />
          <meshStandardMaterial color={0x4a4a4a} roughness={0.8} />
        </mesh>
      </RigidBody>

      {/* Street/Ground texture variation */}
      <mesh
        rotation={[-Math.PI / 2, 0, 0]}
        position={[0, 0.01, 0]}
        receiveShadow
      >
        <planeGeometry args={[200, 8]} />
        <meshStandardMaterial color={0x2a2a2a} roughness={0.9} />
      </mesh>

      {/* Building 1 - Left side, tall */}
      <RigidBody type="fixed" colliders="cuboid">
        <group position={[-30, 0, -5]}>
          <mesh castShadow receiveShadow>
            <boxGeometry args={[8, 20, 10]} />
            <meshStandardMaterial color={0x3a3a3a} roughness={0.7} />
          </mesh>
          {/* Windows */}
          {[0, 1, 2, 3].map((i) => (
            <mesh key={i} position={[0, -8 + i * 4, 5.01]} castShadow>
              <boxGeometry args={[1.5, 1.5, 0.1]} />
              <meshStandardMaterial
                color={0xffff88}
                emissive={0x332200}
                emissiveIntensity={0.3}
              />
            </mesh>
          ))}
        </group>
      </RigidBody>

      {/* Building 2 - Left side, medium */}
      <RigidBody type="fixed" colliders="cuboid">
        <group position={[-15, 0, -8]}>
          <mesh castShadow receiveShadow>
            <boxGeometry args={[6, 15, 8]} />
            <meshStandardMaterial color={0x4a4a4a} roughness={0.7} />
          </mesh>
          {/* Windows */}
          {[0, 1, 2].map((i) => (
            <mesh key={i} position={[0, -6 + i * 4, 4.01]} castShadow>
              <boxGeometry args={[1.2, 1.2, 0.1]} />
              <meshStandardMaterial
                color={0xffff88}
                emissive={0x332200}
                emissiveIntensity={0.2}
              />
            </mesh>
          ))}
        </group>
      </RigidBody>

      {/* Building 3 - Right side, tall with different style */}
      <RigidBody type="fixed" colliders="cuboid">
        <group position={[30, 0, -6]}>
          <mesh castShadow receiveShadow>
            <boxGeometry args={[10, 25, 12]} />
            <meshStandardMaterial color={0x2a2a2a} roughness={0.6} />
          </mesh>
          {/* Windows grid */}
          {[0, 1, 2, 3, 4].map((row) => (
            <group key={row} position={[0, -10 + row * 4, 6.01]}>
              {[-2, 0, 2].map((col) => (
                <mesh key={col} position={[col, 0, 0]} castShadow>
                  <boxGeometry args={[1, 1, 0.1]} />
                  <meshStandardMaterial
                    color={0x88ccff}
                    emissive={0x001122}
                    emissiveIntensity={0.4}
                  />
                </mesh>
              ))}
            </group>
          ))}
        </group>
      </RigidBody>

      {/* Building 4 - Right side, medium */}
      <RigidBody type="fixed" colliders="cuboid">
        <group position={[15, 0, -7]}>
          <mesh castShadow receiveShadow>
            <boxGeometry args={[7, 18, 9]} />
            <meshStandardMaterial color={0x5a5a5a} roughness={0.7} />
          </mesh>
          {/* Windows */}
          {[0, 1, 2, 3].map((i) => (
            <mesh key={i} position={[0, -7.5 + i * 4, 4.51]} castShadow>
              <boxGeometry args={[1.3, 1.3, 0.1]} />
              <meshStandardMaterial
                color={0xffaa88}
                emissive={0x331100}
                emissiveIntensity={0.3}
              />
            </mesh>
          ))}
        </group>
      </RigidBody>

      {/* Building 5 - Center, short */}
      <RigidBody type="fixed" colliders="cuboid">
        <group position={[0, 0, -4]}>
          <mesh castShadow receiveShadow>
            <boxGeometry args={[5, 12, 6]} />
            <meshStandardMaterial color={0x6a6a6a} roughness={0.7} />
          </mesh>
          {/* Windows */}
          {[0, 1, 2].map((i) => (
            <mesh key={i} position={[0, -4 + i * 4, 3.01]} castShadow>
              <boxGeometry args={[1, 1, 0.1]} />
              <meshStandardMaterial
                color={0xffffff}
                emissive={0x222222}
                emissiveIntensity={0.5}
              />
            </mesh>
          ))}
        </group>
      </RigidBody>

      {/* Fence/Wall elements */}
      {[-25, 25].map((x) => (
        <RigidBody key={x} type="fixed" colliders="cuboid">
          <mesh position={[x, 2, -2]} castShadow receiveShadow>
            <boxGeometry args={[1, 4, 1]} />
            <meshStandardMaterial color={0x8b7355} roughness={0.8} />
          </mesh>
        </RigidBody>
      ))}

      {/* Background fog/depth elements - far buildings */}
      <group position={[0, 0, -20]}>
        {[-40, -20, 20, 40].map((x) => (
          <mesh key={x} position={[x, 0, 0]} castShadow receiveShadow>
            <boxGeometry args={[12, 30, 8]} />
            <meshStandardMaterial
              color={0x1a1a1a}
              roughness={0.8}
              opacity={0.6}
              transparent
            />
          </mesh>
        ))}
      </group>

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

      {/* Stairs - Progressive steps */}
      {[0, 1, 2, 3, 4].map((i) => (
        <RigidBody key={`stair-${i}`} type="fixed" colliders="cuboid">
          <mesh
            position={[35 + i * 1.2, 0.5 + i * 0.8, 0]}
            castShadow
            receiveShadow
          >
            <boxGeometry args={[1.2, 0.5 + i * 0.8, 3]} />
            <meshStandardMaterial color={0x8b6f47} roughness={0.9} />
          </mesh>
        </RigidBody>
      ))}

      {/* Small valley/gap - ground with a gap for jumping practice */}
      <RigidBody type="fixed" colliders="cuboid">
        <mesh position={[50, 0, 0]} castShadow receiveShadow>
          <boxGeometry args={[8, 1, 3]} />
          <meshStandardMaterial color={0x4a4a4a} roughness={0.8} />
        </mesh>
      </RigidBody>

      <RigidBody type="fixed" colliders="cuboid">
        <mesh position={[63, 0, 0]} castShadow receiveShadow>
          <boxGeometry args={[8, 1, 3]} />
          <meshStandardMaterial color={0x4a4a4a} roughness={0.8} />
        </mesh>
      </RigidBody>

      {/* Elevated platforms at different heights */}
      <RigidBody type="fixed" colliders="cuboid">
        <mesh position={[70, 2, 0]} castShadow receiveShadow>
          <boxGeometry args={[3, 0.5, 3]} />
          <meshStandardMaterial color={0x7a7a7a} roughness={0.8} />
        </mesh>
      </RigidBody>

      <RigidBody type="fixed" colliders="cuboid">
        <mesh position={[75, 4, 0]} castShadow receiveShadow>
          <boxGeometry args={[3, 0.5, 3]} />
          <meshStandardMaterial color={0x7a7a7a} roughness={0.8} />
        </mesh>
      </RigidBody>

      <RigidBody type="fixed" colliders="cuboid">
        <mesh position={[80, 6, 0]} castShadow receiveShadow>
          <boxGeometry args={[3, 0.5, 3]} />
          <meshStandardMaterial color={0x7a7a7a} roughness={0.8} />
        </mesh>
      </RigidBody>

      {/* Ramp/Slope */}
      <RigidBody type="fixed" colliders="cuboid">
        <mesh
          position={[90, 2, 0]}
          rotation={[0, 0, -0.3]}
          castShadow
          receiveShadow
        >
          <boxGeometry args={[8, 0.5, 3]} />
          <meshStandardMaterial color={0x6a5a3a} roughness={0.9} />
        </mesh>
      </RigidBody>

      {/* Dynamic physics objects - Pushable cubes (small, character can push) */}
      {[-3, 2, 8, 12].map((x, i) => (
        <RigidBody
          key={`cube-${i}`}
          colliders="cuboid"
          position={[x, 0.15, 0]}
          friction={0.8}
          restitution={0.2}
          mass={0.5}
        >
          <mesh castShadow receiveShadow>
            <boxGeometry args={[0.3, 0.3, 0.3]} />
            <meshStandardMaterial color={0xff6b6b} roughness={0.5} />
          </mesh>
        </RigidBody>
      ))}

      {/* Dynamic physics objects - Pushable spheres (light and bouncy) */}
      {[25, 28, 32].map((x, i) => (
        <RigidBody
          key={`sphere-${i}`}
          colliders="ball"
          position={[x, 0.2, 0]}
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
      ))}

      {/* Medium pushable crate (heavier but still pushable) */}
      <RigidBody
        colliders="cuboid"
        position={[45, 0.3, 0]}
        friction={0.9}
        restitution={0.1}
        mass={1.5}
      >
        <mesh castShadow receiveShadow>
          <boxGeometry args={[0.6, 0.6, 0.6]} />
          <meshStandardMaterial color={0x8b5a00} roughness={0.9} />
        </mesh>
      </RigidBody>

      {/* Grass component */}
      {(() => {
        // Controls are flat, not nested (Leva flattens folders)
        const enabled = grassControls?.enabled ?? false;

        if (!enabled) return null;

        // Access controls directly from root (they're flattened by Leva)
        const cgq3 = grassControls || {};

        return (
          <ClaudeGrassQuick3
            key="grass"
            playerPosition={playerPosRef.current}
            terrainSize={cgq3.terrainSize ?? 100}
            heightScale={cgq3.heightScale ?? 1}
            heightOffset={cgq3.heightOffset ?? 0}
            grassWidth={cgq3.grassWidth ?? 0.02}
            grassHeight={cgq3.grassHeight ?? 0.25}
            lodDistance={cgq3.lodDistance ?? 15}
            maxDistance={cgq3.maxDistance ?? 100}
            patchSize={cgq3.patchSize ?? 10}
            gridSize={cgq3.gridSize ?? 16}
            patchSpacing={cgq3.patchSpacing ?? 10}
            windEnabled={cgq3.windEnabled ?? true}
            windStrength={cgq3.windStrength ?? 1.25}
            windDirectionScale={cgq3.windDirectionScale ?? 0.05}
            windDirectionSpeed={cgq3.windDirectionSpeed ?? 0.05}
            windStrengthScale={cgq3.windStrengthScale ?? 0.25}
            windStrengthSpeed={cgq3.windStrengthSpeed ?? 1.0}
            playerInteractionEnabled={cgq3.playerInteractionEnabled ?? true}
            playerInteractionRepel={cgq3.playerInteractionRepel ?? true}
            playerInteractionRange={cgq3.playerInteractionRange ?? 2.5}
            playerInteractionStrength={cgq3.playerInteractionStrength ?? 0.2}
            playerInteractionHeightThreshold={
              cgq3.playerInteractionHeightThreshold ?? 3.0
            }
            baseColor1={cgq3.baseColor1 ?? "#051303"}
            baseColor2={cgq3.baseColor2 ?? "#061a03"}
            tipColor1={cgq3.tipColor1 ?? "#a6cc40"}
            tipColor2={cgq3.tipColor2 ?? "#cce666"}
            gradientCurve={cgq3.gradientCurve ?? 4.0}
            aoEnabled={cgq3.aoEnabled ?? true}
            aoIntensity={cgq3.aoIntensity ?? 1.0}
            grassMiddleBrightnessMin={cgq3.grassMiddleBrightnessMin ?? 0.85}
            grassMiddleBrightnessMax={cgq3.grassMiddleBrightnessMax ?? 1.0}
            fogEnabled={cgq3.fogEnabled ?? false}
            fogNear={cgq3.fogNear ?? 5.0}
            fogFar={cgq3.fogFar ?? 50.0}
            fogIntensity={cgq3.fogIntensity ?? 1.0}
            fogColor={cgq3.fogColor ?? "#4f74af"}
            specularEnabled={cgq3.specularEnabled ?? false}
            specularIntensity={cgq3.specularIntensity ?? 2.0}
            specularColor={cgq3.specularColor ?? "#ffffff"}
            specularDirectionX={cgq3.specularDirectionX ?? -1.0}
            specularDirectionY={cgq3.specularDirectionY ?? 1.0}
            specularDirectionZ={cgq3.specularDirectionZ ?? 0.5}
            backscatterEnabled={cgq3.backscatterEnabled ?? true}
            backscatterIntensity={cgq3.backscatterIntensity ?? 0.5}
            backscatterColor={cgq3.backscatterColor ?? "#51cc66"}
            backscatterPower={cgq3.backscatterPower ?? 2.0}
            frontScatterStrength={cgq3.frontScatterStrength ?? 0.3}
            rimSSSStrength={cgq3.rimSSSStrength ?? 0.5}
            grassDensity={cgq3.grassDensity ?? 3072}
          />
        );
      })()}
    </group>
  );
};
