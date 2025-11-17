import { useAnimations, useGLTF } from "@react-three/drei";
import React, { useEffect, useRef, useMemo } from "react";
import { useFrame, useGraph } from "@react-three/fiber";
import { useControls, folder } from "leva";
import * as THREE from "three";
import { SkeletonUtils } from "three-stdlib";

interface GodotCharacterProps {
  animation?: string;
  onFootBonesReady?: (bones: {
    leftFoot: THREE.Object3D | null;
    rightFoot: THREE.Object3D | null;
  }) => void;
  [key: string]: any;
}

export function GodotCharacter({
  animation,
  onFootBonesReady,
  ...props
}: GodotCharacterProps) {
  const group = useRef<THREE.Group>(null);
  const animationGroup = useRef<THREE.Group>(null);
  const bonesNotifiedRef = useRef(false);

  // Load the model
  const { scene, animations } = useGLTF(
    "/models/AnimationLibrary_Godot_Standard-transformed.glb"
  );

  // Clone the scene properly with skeleton
  const clone = useMemo(() => SkeletonUtils.clone(scene), [scene]);
  const { nodes, materials } = useGraph(clone) as any;

  const { actions, mixer } = useAnimations(animations, animationGroup);

  // Character controls
  const { yPosition, showDebug, characterScaleDisplay } = useControls(
    "🎮 GODOT CHARACTER",
    {
      characterVisuals: folder(
        {
          yPosition: {
            value: -0.99,
            min: -2,
            max: 2,
            step: 0.01,
            label: "Feet Position - Align feet with capsule bottom",
          },
          characterScaleDisplay: {
            value: 1,
            min: 0.1,
            max: 5,
            step: 0.1,
            label: "Character Visual Scale (1 = realistic)",
          },
          showDebug: {
            value: false,
            label: "Show Alignment Markers (Orange=Ground)",
          },
        },
        { collapsed: true }
      ),
    }
  );

  // Use the character scale directly (no conversion needed)
  const characterScale = characterScaleDisplay;

  // Validate character setup once
  useEffect(() => {
    // Check character bounding box for real-world size
    if (nodes.Mannequin_1) {
      nodes.Mannequin_1.geometry.computeBoundingBox();
    }
  }, [nodes]);

  useEffect(() => {
    bonesNotifiedRef.current = false;
  }, [onFootBonesReady]);

  useEffect(() => {
    if (!nodes || bonesNotifiedRef.current) {
      return;
    }

    const footNodeNames = Object.keys(nodes).filter((name) =>
      name.toLowerCase().includes("foot")
    );

    if (footNodeNames.length > 0) {
      console.log("GodotCharacter foot-related nodes:", footNodeNames);
    } else {
      console.warn("GodotCharacter: no foot nodes detected in GLB graph.");
    }
  }, [nodes, bonesNotifiedRef]);

  useEffect(() => {
    if (
      bonesNotifiedRef.current ||
      !onFootBonesReady ||
      !animationGroup.current
    ) {
      return;
    }

    const findBoneByCandidates = (candidates: string[]) => {
      for (const candidate of candidates) {
        const directMatch = nodes[candidate];
        if (directMatch) {
          return directMatch;
        }
      }

      const lowerCandidates = candidates.map((name) => name.toLowerCase());

      return Object.values(nodes).find((node: any) => {
        if (!node || typeof node !== "object") return false;
        if (!("isBone" in node) || !node.isBone) return false;
        const nodeName = (node.name || "").toLowerCase();
        return lowerCandidates.some((candidate) =>
          nodeName.includes(candidate.toLowerCase())
        );
      }) as THREE.Object3D | undefined;
    };

    const leftFootCandidates = [
      "mixamorigLeftFoot",
      "mixamorig_LeftFoot",
      "LeftFoot",
      "DEF-footL",
    ];
    const rightFootCandidates = [
      "mixamorigRightFoot",
      "mixamorig_RightFoot",
      "RightFoot",
      "DEF-footR",
    ];

    const leftFoot =
      (findBoneByCandidates(leftFootCandidates) as
        | THREE.Object3D
        | undefined) || null;
    const rightFoot =
      (findBoneByCandidates(rightFootCandidates) as
        | THREE.Object3D
        | undefined) || null;

    onFootBonesReady({
      leftFoot,
      rightFoot,
    });

    bonesNotifiedRef.current = true;
  }, [nodes, onFootBonesReady]);

  // Animation mapping - map our animation names to Godot animation names
  const animationMap: { [key: string]: string } = {
    idle: "Idle_Loop",
    walk: "Walk_Loop",
    run: "Sprint_Loop",
    walkBackwards: "Walk_Loop", // No backwards walk, use regular walk
    leftTurn: "Walk_Loop",
    rightTurn: "Walk_Loop",
    dance: "Dance_Loop",
    jumpStart: "Jump_Start",
    jumpLoop: "Jump_Loop",
    jumpLand: "Jump_Land",
    crouchIdle: "Crouch_Idle_Loop",
    crouchWalk: "Crouch_Fwd_Loop",
    // Combat animations
    swordIdle: "Sword_Idle",
    swordAttack: "Sword_Attack",
    swordAttackAlt: "Sword_Attack_RM",
    roll: "Roll",
  };

  const currentAnimationRef = useRef<string | null>(null);

  // Force reset position every frame to prevent sliding
  useFrame(() => {
    if (animationGroup.current) {
      animationGroup.current.position.set(0, yPosition, 0);
      animationGroup.current.rotation.set(0, 0, 0);
    }
  });

  useEffect(() => {
    if (actions && animation && animationGroup.current) {
      const mappedAnimation = animationMap[animation] || animationMap.idle;

      if (currentAnimationRef.current !== mappedAnimation) {
        const previousAction = currentAnimationRef.current
          ? actions[currentAnimationRef.current]
          : null;
        const nextAction = actions[mappedAnimation];

        if (nextAction) {
          // If there was a previous animation, fade it out
          if (previousAction) {
            previousAction.fadeOut(0.2);
          }

          // Fade in the new animation
          nextAction
            .reset()
            .setEffectiveTimeScale(1)
            .setEffectiveWeight(1)
            .fadeIn(0.2)
            .play();

          currentAnimationRef.current = mappedAnimation;
        } else {
          // Fallback to Idle_Loop if mapping doesn't exist
          if (actions["Idle_Loop"]) {
            if (previousAction) {
              previousAction.fadeOut(0.2);
            }
            actions["Idle_Loop"].reset().fadeIn(0.2).play();
            currentAnimationRef.current = "Idle_Loop";
          }
        }
      }
    }
  }, [animation, actions, animationMap]);

  return (
    <group ref={group} {...props}>
      {/* Visual Debug markers */}
      {showDebug && (
        <>
          {/* Character center marker (chest level) */}
          <mesh position={[0, yPosition + 0.9, 0]}>
            <boxGeometry args={[0.1, 0.1, 0.1]} />
            <meshBasicMaterial color="blue" />
          </mesh>

          {/* Character feet marker */}
          <mesh position={[0, yPosition, 0]}>
            <sphereGeometry args={[0.05]} />
            <meshBasicMaterial color="green" />
          </mesh>

          {/* Ground level marker (0,0,0) */}
          <mesh position={[0, 0, 0]}>
            <cylinderGeometry args={[0.2, 0.2, 0.02]} />
            <meshBasicMaterial color="orange" />
          </mesh>
        </>
      )}

      {/* Animation group */}
      <group
        ref={animationGroup}
        position={[0, yPosition, 0]}
        scale={characterScale}
      >
        <group name="Scene">
          <group name="Rig">
            {nodes.root && <primitive object={nodes.root} />}
          </group>
          <group name="Mannequin">
            {nodes.Mannequin_1 && (
              <skinnedMesh
                name="Mannequin_1"
                geometry={nodes.Mannequin_1.geometry}
                material={materials?.M_Main || nodes.Mannequin_1.material}
                skeleton={nodes.Mannequin_1.skeleton}
                castShadow
                receiveShadow
                frustumCulled={false}
              />
            )}
            {nodes.Mannequin_2 && (
              <skinnedMesh
                name="Mannequin_2"
                geometry={nodes.Mannequin_2.geometry}
                material={materials?.M_Joints || nodes.Mannequin_2.material}
                skeleton={nodes.Mannequin_2.skeleton}
                castShadow
                receiveShadow
                frustumCulled={false}
              />
            )}
          </group>
        </group>
      </group>
    </group>
  );
}

useGLTF.preload("/models/AnimationLibrary_Godot_Standard-transformed.glb");
