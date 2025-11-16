import { useKeyboardControls } from "@react-three/drei";
import { useFrame } from "@react-three/fiber";
import { CapsuleCollider, RigidBody, useRapier } from "@react-three/rapier";
import { useControls } from "leva";
import { useRef, useState } from "react";
import { MathUtils } from "three";
import { Character } from "./Character";

const normalizeAngle = (angle) => {
  while (angle > Math.PI) angle -= 2 * Math.PI;
  while (angle < -Math.PI) angle += 2 * Math.PI;
  return angle;
};

const lerpAngle = (start, end, t) => {
  start = normalizeAngle(start);
  end = normalizeAngle(end);

  if (Math.abs(end - start) > Math.PI) {
    if (end > start) {
      start += 2 * Math.PI;
    } else {
      end += 2 * Math.PI;
    }
  }

  return normalizeAngle(start + (end - start) * t);
};

export const CharacterController = ({
  mapType = "glb",
  enableOrbitControls = false,
  playerPositionRef,
  setPlayerPosition,
}) => {
  const { WALK_SPEED, RUN_SPEED, ROTATION_SPEED, JUMP_FORCE } = useControls(
    "Character Control",
    {
      WALK_SPEED: { value: 0.8, min: 0.1, max: 4, step: 0.1 },
      RUN_SPEED: { value: 1.6, min: 0.2, max: 12, step: 0.1 },
      ROTATION_SPEED: {
        value: MathUtils.degToRad(0.5),
        min: MathUtils.degToRad(0.1),
        max: MathUtils.degToRad(5),
        step: MathUtils.degToRad(0.1),
      },
      JUMP_FORCE: { value: 5, min: 1, max: 15, step: 0.5, label: "Jump Force" },
    }
  );

  const {
    cameraOffsetX,
    cameraOffsetY,
    cameraOffsetZ,
    cameraTargetX,
    cameraTargetY,
    cameraTargetZ,
    enableCameraLerp,
    cameraLerpFactor,
  } = useControls("Camera Position (2.5D Side View)", {
    cameraOffsetX: {
      value: 0,
      min: -10,
      max: 10,
      step: 0.1,
      label: "Camera X Offset (Left/Right)",
    },
    cameraOffsetY: {
      value: 0.6,
      min: 0,
      max: 20,
      step: 0.1,
      label: "Camera Y Offset (Height)",
    },
    cameraOffsetZ: {
      value: 4.4,
      min: 1,
      max: 50,
      step: 0.1,
      label: "Camera Z Distance (Side View Depth)",
    },
    cameraTargetX: {
      value: 0,
      min: -10,
      max: 10,
      step: 0.1,
      label: "Look At X Offset",
    },
    cameraTargetY: {
      value: 0.2,
      min: -5,
      max: 10,
      step: 0.1,
      label: "Look At Y Offset",
    },
    cameraTargetZ: {
      value: 0,
      min: -5,
      max: 5,
      step: 0.1,
      label: "Look At Z Offset",
    },
    enableCameraLerp: {
      value: true,
      label: "Enable Camera Lerp (Smooth Follow)",
    },
    cameraLerpFactor: {
      value: 0.1,
      min: 0.01,
      max: 1,
      step: 0.01,
      label: "Camera Lerp Factor",
    },
  });

  const { capsuleHalfHeight, capsuleRadius } = useControls("Capsule Collider", {
    capsuleHalfHeight: {
      value: 0.12,
      min: 0.05,
      max: 0.5,
      step: 0.01,
      label: "Half Height",
    },
    capsuleRadius: {
      value: 0.15,
      min: 0.05,
      max: 0.5,
      step: 0.01,
      label: "Radius",
    },
  });
  const rb = useRef();
  const container = useRef();
  const character = useRef();

  const [animation, setAnimation] = useState("idle");

  const characterRotationTarget = useRef(0); // 0 = right, Math.PI = left
  const [, get] = useKeyboardControls();
  const { rapier, world } = useRapier();
  const isOnGround = useRef(false);
  const jumpPressed = useRef(false);

  useFrame(({ camera }) => {
    if (rb.current) {
      const vel = rb.current.linvel();
      const pos = rb.current.translation();

      // Ground check - raycast downward from bottom of capsule
      const rayOrigin = { x: pos.x, y: pos.y - 0.1, z: pos.z };
      const rayDir = { x: 0, y: -1, z: 0 };
      const ray = new rapier.Ray(rayOrigin, rayDir);
      const maxRayDistance = capsuleHalfHeight + capsuleRadius + 0.05; // Capsule bottom + small margin
      const hit = world.castRay(ray, maxRayDistance, true);
      isOnGround.current = hit !== null;

      // Jump logic
      const controls = get();
      const isJumpPressed = controls.jump;

      if (isJumpPressed && !jumpPressed.current && isOnGround.current) {
        // Set upward velocity directly for jump (more reliable than impulse for character controllers)
        vel.y = JUMP_FORCE;
        jumpPressed.current = true;
      }

      // Track jump button release
      if (!isJumpPressed) {
        jumpPressed.current = false;
      }

      // 2.5D Side-scroller: Only move in X (left/right), no Y movement (gravity handles that)
      let movementX = 0;

      if (controls.left) {
        movementX = -1; // Left
      }
      if (controls.right) {
        movementX = 1; // Right
      }

      let speed = controls.run ? RUN_SPEED : WALK_SPEED;

      // Apply movement only in X (2D plane)
      // Don't override Y velocity - let gravity work
      if (movementX !== 0) {
        vel.x = movementX * speed;

        // Face the direction of movement (left or right)
        // For side-scroller: right = Math.PI, left = 0
        characterRotationTarget.current = movementX > 0 ? Math.PI : 0;

        if (speed === RUN_SPEED) {
          setAnimation("run");
        } else {
          setAnimation("walk");
        }
      } else {
        vel.x = 0;
        // Don't set vel.y = 0, let gravity work
        if (isOnGround.current) {
          setAnimation("idle");
        }
      }

      // Lock Z velocity to prevent depth movement (2.5D)
      vel.z = 0;

      // Rotate character to face movement direction
      // Add -Math.PI/2 to make character face camera in side-scroller view
      const baseRotation = -Math.PI / 2; // Face camera
      character.current.rotation.y = lerpAngle(
        character.current.rotation.y,
        baseRotation + characterRotationTarget.current,
        0.1
      );

      // Lock Z position to prevent moving into/out of screen
      const currentPos = rb.current.translation();
      if (Math.abs(currentPos.z) > 0.01) {
        rb.current.setTranslation(
          { x: currentPos.x, y: currentPos.y, z: 0 },
          true
        );
      }

      rb.current.setLinvel(vel, true);
    }

    // Update player position ref for grass component
    if (rb.current) {
      const playerPos = rb.current.translation();
      const posArray = [playerPos.x, playerPos.y, playerPos.z];
      if (playerPositionRef) {
        playerPositionRef.current = posArray;
      }
      if (setPlayerPosition) {
        setPlayerPosition(posArray);
      }
    }

    // CAMERA - Only control camera if OrbitControls is disabled
    // 2.5D Side-scroller: Camera follows character in X and Y, fixed side-view angle
    if (!enableOrbitControls && rb.current) {
      const playerPos = rb.current.translation();

      // Camera follows player in X and Y, but stays at fixed Z (side view)
      const targetCameraX = playerPos.x + cameraOffsetX;
      const targetCameraY = playerPos.y + cameraOffsetY;
      const fixedCameraZ = cameraOffsetZ; // Fixed depth for side view

      if (enableCameraLerp) {
        camera.position.x = MathUtils.lerp(
          camera.position.x,
          targetCameraX,
          cameraLerpFactor
        );
        camera.position.y = MathUtils.lerp(
          camera.position.y,
          targetCameraY,
          cameraLerpFactor
        );
      } else {
        camera.position.x = targetCameraX;
        camera.position.y = targetCameraY;
      }
      camera.position.z = fixedCameraZ; // Always locked to side view position

      // Look at player position with offset
      camera.lookAt(
        playerPos.x + cameraTargetX,
        playerPos.y + cameraTargetY,
        playerPos.z + cameraTargetZ
      );
    }
  });

  // Spawn higher for procedural terrain to ensure visibility
  // For side-scroller, spawn on ground level, avoiding street lamps at x=-20,-10,0,10,20
  const spawnPosition =
    mapType === "procedural"
      ? [0, 10, 0]
      : mapType === "sidescroller" || mapType === "hillysidescroller"
      ? [-5, 1, 0] // Spawn at x=-5 to avoid lamp at x=0
      : [0, 0, 0];

  return (
    <RigidBody
      colliders={false}
      lockRotations
      lockTranslations={[false, false, true]} // Lock Z translation (depth)
      ref={rb}
      position={spawnPosition}
      friction={0.5}
      restitution={0}
      linearDamping={0.5} // Reduced from 2 - less damping for better jumping
      angularDamping={1}
      ccd={true} // Continuous Collision Detection for fast movement
      canSleep={false} // Prevent player from sleeping (important for controls)
    >
      <group ref={container}>
        <group ref={character}>
          <Character scale={0.18} position-y={-0.25} animation={animation} />
        </group>
      </group>
      <CapsuleCollider
        args={[capsuleHalfHeight, capsuleRadius]}
        friction={0.7}
      />
    </RigidBody>
  );
};
