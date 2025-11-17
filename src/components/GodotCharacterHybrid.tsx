// Hybrid approach: Rapier for physics + BVH for ground detection
import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useKeyboardControls } from "@react-three/drei";
import { useFrame, useThree } from "@react-three/fiber";
import { CapsuleCollider, RigidBody, useRapier } from "@react-three/rapier";
import { useControls, folder } from "leva";
import { MathUtils, Vector3, Matrix4, Line3, Box3 } from "three";
import { degToRad } from "three/src/math/MathUtils.js";
import { GodotCharacter } from "./GodotCharacter";
import type * as THREE from "three";
import { TeleportationRequest } from "../types/teleportation";
import {
  FootstepParticles,
  type FootstepParticlesHandle,
  type FootstepParticleSpawnOptions,
} from "./FootstepParticles";

const normalizeAngle = (angle: number) => {
  while (angle > Math.PI) angle -= 2 * Math.PI;
  while (angle < -Math.PI) angle += 2 * Math.PI;
  return angle;
};

const lerpAngle = (start: number, end: number, t: number) => {
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

interface Props {
  position?: [number, number, number];
  cameraMode?: string;
  collider?: THREE.Mesh | null;
  onPositionChange?: (position: [number, number, number]) => void;
  onVelocityChange?: (velocity: [number, number, number]) => void;
  onRotationChange?: (rotation: number) => void;
  teleportRequest?: TeleportationRequest | null;
  onTeleportHandled?: (id: string) => void;
}

export const GodotCharacterHybrid = ({
  position = [0, 2, 0],
  cameraMode = "orbit",
  collider = null,
  onPositionChange,
  onVelocityChange,
  onRotationChange,
  teleportRequest,
  onTeleportHandled,
}: Props) => {
  // Access Rapier world for raycasting dynamic objects
  const { world, rapier } = useRapier();
  const { camera } = useThree();

  const {
    WALK_SPEED,
    RUN_SPEED,
    ROTATION_SPEED,
    JUMP_FORCE,
    cameraX,
    cameraY,
    cameraZ,
    targetZ,
    cameraLerpSpeed,
    mouseSensitivity,
    capsuleHeight,
    capsuleRadius,
    enableFootstepAudio,
    enableFootstepParticles,
  } = useControls("🎮 GODOT CHARACTER", {
    control: folder(
      {
        WALK_SPEED: { value: 1.8, min: 0.1, max: 4, step: 0.1 },
        RUN_SPEED: { value: 4, min: 0.2, max: 12, step: 0.1 },
        ROTATION_SPEED: {
          value: degToRad(0.5),
          min: degToRad(0.1),
          max: degToRad(5),
          step: degToRad(0.1),
        },
        JUMP_FORCE: { value: 6, min: 1, max: 10, step: 0.1 },
      },
      { collapsed: true }
    ),
    camera: folder(
      {
        cameraX: { value: 0, min: -10, max: 10, step: 0.1 },
        cameraY: { value: 1.5, min: 0, max: 10, step: 0.1 },
        cameraZ: { value: -5.6, min: -10, max: 2, step: 0.1 },
        targetZ: { value: 5, min: -2, max: 5, step: 0.1 },
        cameraLerpSpeed: { value: 0.1, min: 0.01, max: 0.5, step: 0.01 },
        mouseSensitivity: {
          value: 0.003,
          min: 0.001,
          max: 0.01,
          step: 0.001,
          label: "Mouse Sensitivity",
        },
      },
      { collapsed: true }
    ),
    capsule: folder(
      {
        capsuleHeight: {
          value: 1.4,
          min: 0.5,
          max: 2.0,
          step: 0.05,
          label: "Capsule Total Height",
        },
        capsuleRadius: {
          value: 0.3,
          min: 0.05,
          max: 0.3,
          step: 0.01,
          label: "Capsule Radius",
        },
      },
      { collapsed: true }
    ),
    footsteps: folder(
      {
        enableFootstepAudio: {
          value: true,
          label: "Enable Footstep Audio",
        },
        enableFootstepParticles: {
          value: true,
          label: "Enable Footstep Particles",
        },
      },
      { collapsed: true }
    ),
  });

  const rb = useRef<any>(null);
  const container = useRef<any>(null);
  const character = useRef<any>(null);
  const [animation, setAnimation] = useState("idle");
  const animationRef = useRef(animation);
  useEffect(() => {
    animationRef.current = animation;
  }, [animation]);
  const [isGrounded, setIsGrounded] = useState(true);
  const wasGrounded = useRef(false);
  const jumpPhase = useRef<"none" | "start" | "loop" | "land">("none");
  const [combatMode, setCombatMode] = useState(false);
  const isAttacking = useRef(false);
  const isRolling = useRef(false);
  const rollPressed = useRef(false);

  const characterRotationTarget = useRef(0);
  const rotationTarget = useRef(0);
  const cameraTarget = useRef<any>(null);
  const cameraPosition = useRef<any>(null);
  const cameraWorldPosition = useRef(new Vector3());
  const cameraLookAtWorldPosition = useRef(new Vector3());
  const smoothCameraPosition = useRef(new Vector3());
  const currentCameraPosition = useRef(new Vector3());
  const idealCameraPosition = useRef(new Vector3());
  const smoothLookAtPosition = useRef(new Vector3());
  const cameraLookAt = useRef(new Vector3());
  const [, get] = useKeyboardControls();
  const jumpPressed = useRef(false);
  const cameraInitialized = useRef(false);
  const isCrouchingRef = useRef(false);
  const crouchTransitioningRef = useRef(false);
  const ceilingClearanceTimer = useRef(0);
  const teleportTimeoutRef = useRef<number | null>(null);
  const lastTeleportIdRef = useRef<string | null>(null);
  const teleportHoldFramesRef = useRef(0);
  const teleportCameraPositionRef = useRef(new Vector3());
  const teleportLookAtRef = useRef(new Vector3());

  const footstepParticlesRef = useRef<FootstepParticlesHandle | null>(null);
  const leftFootBone = useRef<THREE.Object3D | null>(null);
  const rightFootBone = useRef<THREE.Object3D | null>(null);
  const leftFootWorldPosition = useRef(new Vector3());
  const rightFootWorldPosition = useRef(new Vector3());
  const prevLeftFootPosition = useRef(new Vector3());
  const prevRightFootPosition = useRef(new Vector3());
  const leftFootInitialized = useRef(false);
  const rightFootInitialized = useRef(false);
  const leftFootWasGrounded = useRef(false);
  const rightFootWasGrounded = useRef(false);
  const footstepCooldownRef = useRef(0);
  const footBonesMissingWarnedRef = useRef(false);
  const lastFootstepIndexRef = useRef<number | null>(null);
  const tempLandingCenterRef = useRef(new Vector3());
  const tempLandingNormalRef = useRef(new Vector3(0, 1, 0));
  const leftFootPrevToi = useRef(1);
  const rightFootPrevToi = useRef(1);

  const footstepAnimations = useMemo(
    () => new Set(["walk", "run", "walkBackwards", "crouchWalk"]),
    []
  );

  const footstepSoundPaths = useMemo(
    () => [
      "/sounds/steps.mp3",
      "/sounds/steps (2).mp3",
      "/sounds/steps (3).mp3",
      "/sounds/steps (5).mp3",
    ],
    []
  );

  const playFootstepSound = useCallback(() => {
    if (
      !enableFootstepAudio ||
      typeof window === "undefined" ||
      footstepSoundPaths.length === 0
    ) {
      return;
    }

    let chosenIndex: number;
    if (footstepSoundPaths.length === 1) {
      chosenIndex = 0;
    } else {
      let attempts = 0;
      do {
        chosenIndex = Math.floor(Math.random() * footstepSoundPaths.length);
        attempts += 1;
      } while (chosenIndex === lastFootstepIndexRef.current && attempts < 5);
    }

    lastFootstepIndexRef.current = chosenIndex;
    const clip = footstepSoundPaths[chosenIndex];

    const audio = new Audio(clip);
    audio.volume = 0.3;
    audio.play().catch(() => {
      /* Ignore playback errors (e.g., user gesture requirement) */
    });
  }, [footstepSoundPaths]);

  const castFootRay = useCallback(
    (
      position: THREE.Vector3
    ): (FootstepParticleSpawnOptions & { hitToi: number }) | null => {
      if (!world || !rapier || !rb.current) {
        return null;
      }

      const rayOrigin = {
        x: position.x,
        y: position.y + 0.05,
        z: position.z,
      };
      const rayDirection = { x: 0, y: -1, z: 0 };
      const rayLength = 0.35;

      try {
        const ray = new rapier.Ray(rayOrigin, rayDirection);
        const hit = world.castRayAndGetNormal(
          ray,
          rayLength,
          true,
          undefined,
          undefined,
          undefined,
          rb.current,
          undefined
        );

        if (hit) {
          const hitToi =
            (hit as any).toi ??
            (hit as any).timeOfImpact ??
            (hit as any).time_of_impact ??
            null;

          if (typeof hitToi === "number" && hitToi <= rayLength) {
            const point = new Vector3(
              rayOrigin.x + rayDirection.x * hitToi,
              rayOrigin.y + rayDirection.y * hitToi,
              rayOrigin.z + rayDirection.z * hitToi
            );

            const hitNormal =
              (hit as any).normal ??
              (hit as any).normal1 ??
              (hit as any).normal2 ??
              null;

            let normal: THREE.Vector3;
            if (hitNormal) {
              normal = new Vector3(hitNormal.x, hitNormal.y, hitNormal.z);
            } else {
              normal = new Vector3().copy(tempLandingNormalRef.current);
            }

            const slopeFactor = normal
              ? 1 - Math.max(0, Math.min(1, normal.y))
              : 0;

            return {
              position: point,
              normal,
              slopeFactor,
              hitToi,
            };
          }
        }

        return null;
      } catch (error) {
        console.warn("Footstep raycast error:", error);
        return null;
      }
    },
    [rapier, world]
  );

  const handleFootBonesReady = useCallback(
    (bones: {
      leftFoot: THREE.Object3D | null;
      rightFoot: THREE.Object3D | null;
    }) => {
      leftFootBone.current = bones.leftFoot;
      rightFootBone.current = bones.rightFoot;
      leftFootInitialized.current = false;
      rightFootInitialized.current = false;
      leftFootWasGrounded.current = false;
      rightFootWasGrounded.current = false;
      footBonesMissingWarnedRef.current = false;
    },
    []
  );

  // Mouse orbit for follow-orbit camera mode (delta-based)
  const mouseOrbitOffset = useRef(0); // Horizontal orbit offset (accumulated from mouse movement deltas)
  const mouseVerticalOffset = useRef(0); // Vertical orbit offset (accumulated from mouse movement deltas)
  const isPointerLocked = useRef(false);
  const pointerLockElementRef = useRef<HTMLElement | null>(null);

  // BVH temps
  const tempBox = useRef(new Box3());
  const tempMat = useRef(new Matrix4());
  const tempSegment = useRef(new Line3());
  const tempVector = useRef(new Vector3());
  const tempVector2 = useRef(new Vector3());

  // Combat mode toggle - R key (E is used for dance)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "r" || e.key === "R") {
        setCombatMode((prev) => !prev);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  // Mouse attack handlers
  useEffect(() => {
    const handleMouseDown = (e: MouseEvent) => {
      if (!combatMode || isAttacking.current) return;

      isAttacking.current = true;

      if (e.button === 0) {
        // Left click - primary attack
        setAnimation("swordAttack");
        setTimeout(() => {
          isAttacking.current = false;
        }, 600); // Attack duration
      } else if (e.button === 2) {
        // Right click - secondary attack
        setAnimation("swordAttackAlt");
        setTimeout(() => {
          isAttacking.current = false;
        }, 600);
      }
    };

    const handleContextMenu = (e: MouseEvent) => {
      if (combatMode) {
        e.preventDefault(); // Prevent right-click menu in combat mode
      }
    };

    if (combatMode) {
      window.addEventListener("mousedown", handleMouseDown);
      window.addEventListener("contextmenu", handleContextMenu);
    }

    return () => {
      window.removeEventListener("mousedown", handleMouseDown);
      window.removeEventListener("contextmenu", handleContextMenu);
    };
  }, [combatMode]);

  // Mouse orbit controls for follow-orbit camera mode (delta-based with pointer lock)
  useEffect(() => {
    if (cameraMode !== "follow-orbit") {
      // Unlock pointer and reset if exiting follow-orbit mode
      if (isPointerLocked.current && document.pointerLockElement) {
        document.exitPointerLock();
      }
      return;
    }

    // Get canvas element for pointer lock (React Three Fiber canvas)
    const canvas = document.querySelector("canvas");
    if (!canvas) {
      console.warn("Canvas not found for pointer lock");
      return;
    }
    pointerLockElementRef.current = canvas;

    // Reset offsets when entering follow-orbit mode
    mouseOrbitOffset.current = 0;
    mouseVerticalOffset.current = 0;

    // Request pointer lock on click (required for security)
    const requestPointerLock = () => {
      if (canvas && !isPointerLocked.current) {
        const activeElement = document.activeElement as HTMLElement | null;
        if (
          activeElement &&
          activeElement !== document.body &&
          typeof activeElement.blur === "function"
        ) {
          activeElement.blur();
        }
        canvas.requestPointerLock().catch((err) => {
          console.warn("Pointer lock failed:", err);
        });
      }
    };

    // Handle pointer lock state changes
    const handlePointerLockChange = () => {
      isPointerLocked.current = document.pointerLockElement === canvas;

      // Hide cursor when locked
      if (isPointerLocked.current) {
        const activeElement = document.activeElement as HTMLElement | null;
        if (
          activeElement &&
          activeElement !== document.body &&
          typeof activeElement.blur === "function"
        ) {
          activeElement.blur();
        }
        document.body.style.cursor = "none";
      } else {
        document.body.style.cursor = "auto";
      }
    };

    // Handle mouse movement (delta-based)
    const handleMouseMove = (e: MouseEvent) => {
      if (cameraMode !== "follow-orbit" || !isPointerLocked.current) return;

      // Use movementX and movementY (delta values) instead of absolute position
      // These represent how much the mouse moved since the last event
      const deltaX = e.movementX || 0;
      const deltaY = e.movementY || 0;

      // Accumulate horizontal orbit offset (inverted: right movement = left orbit)
      mouseOrbitOffset.current -= deltaX * mouseSensitivity;

      // Accumulate vertical orbit offset (inverted: up movement = camera looks up)
      mouseVerticalOffset.current -= deltaY * mouseSensitivity;

      // Clamp vertical offset to prevent camera from going too high/low
      mouseVerticalOffset.current = MathUtils.clamp(
        mouseVerticalOffset.current,
        -Math.PI / 3, // ~60 degrees down
        Math.PI / 3 // ~60 degrees up
      );
    };

    // Request pointer lock on click
    canvas.addEventListener("click", requestPointerLock);

    // Listen for pointer lock state changes
    document.addEventListener("pointerlockchange", handlePointerLockChange);
    document.addEventListener("pointerlockerror", () => {
      console.warn("Pointer lock error");
      isPointerLocked.current = false;
      document.body.style.cursor = "auto";
    });

    // Listen for mouse movement (only works when pointer is locked)
    document.addEventListener("mousemove", handleMouseMove);

    // Auto-request pointer lock if not already locked
    if (!isPointerLocked.current) {
      // Small delay to ensure canvas is ready
      setTimeout(() => {
        requestPointerLock();
      }, 100);
    }

    return () => {
      canvas.removeEventListener("click", requestPointerLock);
      document.removeEventListener(
        "pointerlockchange",
        handlePointerLockChange
      );
      document.removeEventListener("mousemove", handleMouseMove);

      // Unlock pointer and restore cursor when cleaning up
      if (document.pointerLockElement === canvas) {
        document.exitPointerLock();
      }
      document.body.style.cursor = "auto";
    };
  }, [cameraMode, mouseSensitivity]);

  useEffect(() => {
    if (!teleportRequest || !rb.current) {
      return;
    }

    if (teleportRequest.id === lastTeleportIdRef.current) {
      return;
    }

    lastTeleportIdRef.current = teleportRequest.id;

    const {
      spawnPosition,
      cameraPosition: cameraTargetPosition,
      lookAtPosition,
      delayMs = 0,
    } = teleportRequest;

    const spawnVec = new Vector3(
      spawnPosition[0],
      spawnPosition[1],
      spawnPosition[2]
    );
    const cameraVec = new Vector3(
      cameraTargetPosition[0],
      cameraTargetPosition[1],
      cameraTargetPosition[2]
    );
    const lookAtVec = new Vector3(
      lookAtPosition[0],
      lookAtPosition[1],
      lookAtPosition[2]
    );

    teleportCameraPositionRef.current.copy(cameraVec);
    teleportLookAtRef.current.copy(lookAtVec);
    teleportHoldFramesRef.current = Math.max(teleportHoldFramesRef.current, 12);

    camera.position.copy(cameraVec);
    camera.lookAt(lookAtVec);
    smoothCameraPosition.current.copy(cameraVec);
    currentCameraPosition.current.copy(cameraVec);
    idealCameraPosition.current.copy(cameraVec);
    smoothLookAtPosition.current.copy(lookAtVec);
    cameraLookAt.current.copy(lookAtVec);
    cameraInitialized.current = true;

    const performTeleport = () => {
      if (!rb.current) {
        return;
      }

      rb.current.setTranslation(
        { x: spawnVec.x, y: spawnVec.y, z: spawnVec.z },
        true
      );
      rb.current.setLinvel({ x: 0, y: 0, z: 0 }, true);
      rb.current.setAngvel({ x: 0, y: 0, z: 0 }, true);

      teleportHoldFramesRef.current = Math.max(
        teleportHoldFramesRef.current,
        8
      );

      jumpPhase.current = "none";
      setAnimation("idle");
      setIsGrounded(true);
      wasGrounded.current = true;

      if (onPositionChange) {
        onPositionChange([spawnVec.x, spawnVec.y, spawnVec.z]);
      }

      if (onVelocityChange) {
        onVelocityChange([0, 0, 0]);
      }

      if (onTeleportHandled) {
        onTeleportHandled(teleportRequest.id);
      }
    };

    if (teleportTimeoutRef.current !== null) {
      window.clearTimeout(teleportTimeoutRef.current);
      teleportTimeoutRef.current = null;
    }

    if (delayMs > 0) {
      teleportTimeoutRef.current = window.setTimeout(() => {
        performTeleport();
        teleportTimeoutRef.current = null;
      }, delayMs);
    } else {
      performTeleport();
    }

    return () => {
      if (teleportTimeoutRef.current !== null) {
        window.clearTimeout(teleportTimeoutRef.current);
        teleportTimeoutRef.current = null;
      }
    };
  }, [
    teleportRequest,
    camera,
    onPositionChange,
    onVelocityChange,
    onTeleportHandled,
    setAnimation,
    setIsGrounded,
  ]);

  // BVH-based ground detection - checks surface normal (STATIC GEOMETRY)
  const checkGroundedBVH = () => {
    if (!rb.current || !collider || !collider.geometry.boundsTree) return false;

    try {
      const position = rb.current.translation();
      const vel = rb.current.linvel();
      if (!vel) return false;

      // Create capsule segment in world space
      const tempSeg = tempSegment.current;
      tempSeg.start.set(position.x, position.y + capsuleRadius, position.z);
      tempSeg.end.set(
        position.x,
        position.y - capsuleHeight / 2 - capsuleRadius,
        position.z
      );

      // Transform to collider local space
      tempMat.current.copy(collider.matrixWorld).invert();
      tempSeg.start.applyMatrix4(tempMat.current);
      tempSeg.end.applyMatrix4(tempMat.current);

      // Create bounding box
      tempBox.current.makeEmpty();
      tempBox.current.expandByPoint(tempSeg.start);
      tempBox.current.expandByPoint(tempSeg.end);
      tempBox.current.min.addScalar(-capsuleRadius);
      tempBox.current.max.addScalar(capsuleRadius);

      let hitGround = false;

      // BVH shapecast to find closest surface
      collider.geometry.boundsTree.shapecast({
        intersectsBounds: (box: any) => box.intersectsBox(tempBox.current),

        intersectsTriangle: (tri: any) => {
          const triPoint = tempVector.current;
          const capsulePoint = tempVector2.current;

          const distance = tri.closestPointToSegment(
            tempSeg,
            triPoint,
            capsulePoint
          );

          if (distance < capsuleRadius + 0.2) {
            // Get triangle normal
            tri.getNormal(tempVector.current);
            const normal = tempVector.current;

            // If normal points up (> 0.7), it's ground
            if (normal.y > 0.7) {
              hitGround = true;
            }
          }

          return false;
        },
      });

      return hitGround;
    } catch (error) {
      console.error("BVH ground check error:", error);
      return false;
    }
  };

  // Rapier raycast for ALL objects (static ground + dynamic cubes)
  const checkGroundedRapier = () => {
    if (!rb.current || !world || !rapier) return false;

    try {
      const position = rb.current.translation();

      // Use current capsule height (accounts for crouch)
      const currentHalfHeight = isCrouchingRef.current
        ? (capsuleHeight * 0.5) / 2
        : capsuleHeight / 2;

      // Cast ray from slightly above character feet downward
      const rayOrigin = {
        x: position.x,
        y: position.y - currentHalfHeight - capsuleRadius + 0.05, // Start just above feet
        z: position.z,
      };
      const rayDirection = { x: 0, y: -1, z: 0 };
      // Short ray - only check immediate ground
      const rayLength = 0.2; // 20cm detection range

      // Create Ray object
      const ray = new rapier.Ray(rayOrigin, rayDirection);

      // Cast ray - EXCLUDE the character's own collider
      // Parameters: ray, maxToi, solid, filterFlags, filterGroups, filterExcludeCollider, filterExcludeRigidBody, filterPredicate
      const hit = world.castRay(
        ray,
        rayLength,
        true, // solid (stop at first hit)
        undefined, // filterFlags
        undefined, // filterGroups
        undefined, // filterExcludeCollider
        rb.current, // filterExcludeRigidBody - EXCLUDE CHARACTER!
        undefined // filterPredicate
      );

      if (hit) {
        const hitToi =
          (hit as any).toi ??
          (hit as any).timeOfImpact ??
          (hit as any).time_of_impact ??
          null;

        // Only count as grounded if hit is within the ray length
        if (typeof hitToi === "number" && hitToi <= rayLength) {
          return true;
        }
      }

      return false;
    } catch (error) {
      console.error("Rapier raycast error:", error);
      return false;
    }
  };

  // Check if there's COMFORTABLE space above to stand up (ceiling detection with buffer)
  const checkCeilingClearance = () => {
    if (!rb.current || !world || !rapier) return true;

    try {
      const position = rb.current.translation();
      const crouchHalfHeight = (capsuleHeight * 0.5) / 2;
      const standingHalfHeight = capsuleHeight / 2;

      // Cast ray UPWARD from top of crouched capsule
      const rayOrigin = {
        x: position.x,
        y: position.y + crouchHalfHeight + capsuleRadius,
        z: position.z,
      };
      const rayDirection = { x: 0, y: 1, z: 0 };
      // Check for COMFORTABLE clearance - larger buffer for safety
      const safetyBuffer = 0.5; // 50cm comfort zone!
      const rayLength = standingHalfHeight - crouchHalfHeight + safetyBuffer;

      const ray = new rapier.Ray(rayOrigin, rayDirection);
      const hit = world.castRay(
        ray,
        rayLength,
        true,
        undefined,
        undefined,
        undefined,
        rb.current,
        undefined
      );

      // If hit something above, NO comfortable clearance to stand
      if (hit) {
        return false;
      }

      return true; // Comfortable clearance - safe to stand!
    } catch (error) {
      console.error("Ceiling check error:", error);
      return true; // Default to allowing stand
    }
  };

  useFrame((_state, delta) => {
    if (rb.current) {
      footstepCooldownRef.current = Math.max(
        footstepCooldownRef.current - delta,
        0
      );

      const vel = rb.current.linvel();
      if (!vel) return;

      // Update position and velocity for external components (like leaves)
      if (onPositionChange) {
        const pos = rb.current.translation();
        onPositionChange([pos.x, pos.y, pos.z]);
      }

      if (onVelocityChange) {
        onVelocityChange([vel.x, vel.y, vel.z]);
      }

      // Get crouch input FIRST (needed for ground detection)
      const crouchInput = get().crouch;

      // SIMPLE GROUND DETECTION - Rapier ONLY (pass current crouch state)
      let grounded = checkGroundedRapier();

      // FORCE grounded during crouch transitions to prevent fall animations
      if (crouchTransitioningRef.current) {
        grounded = true;
      }

      setIsGrounded(grounded);

      // Get other input states
      const danceInput = get().dance;
      const rollInput = get().roll;

      // Handle crouch input and update capsule state FIRST
      // FORCED CROUCH WITH DELAY: Only delay when auto-standing from forced crouch
      // ONLY check ceiling if already crouching (prevents auto-crouch when walking under objects)
      const hasCeilingClearance = isCrouchingRef.current
        ? checkCeilingClearance()
        : true;

      // Timer management for forced crouch
      if (crouchInput) {
        // Manual crouch - disable timer
        ceilingClearanceTimer.current = -1;
      } else if (!hasCeilingClearance && isCrouchingRef.current) {
        // Under ceiling (forced crouch) - set timer to 0 (ready to start)
        ceilingClearanceTimer.current = 0;
      } else if (
        hasCeilingClearance &&
        isCrouchingRef.current &&
        ceilingClearanceTimer.current >= 0
      ) {
        // Has clearance now - start counting
        ceilingClearanceTimer.current += delta;
      } else if (!isCrouchingRef.current) {
        // Standing - reset timer
        ceilingClearanceTimer.current = -1;
      }

      // Stand up delay - ONLY applies when exiting forced crouch
      const standUpDelay = 0.5; // 500ms delay for stability

      let shouldBeCrouched;
      if (crouchInput) {
        // Manual crouch - immediate, no delay
        shouldBeCrouched = true;
        ceilingClearanceTimer.current = -1; // Reset timer
      } else if (!hasCeilingClearance) {
        // Ceiling blocking - forced crouch
        shouldBeCrouched = true;
      } else if (
        ceilingClearanceTimer.current >= 0 &&
        ceilingClearanceTimer.current < standUpDelay
      ) {
        // Exiting forced crouch - apply delay
        shouldBeCrouched = true;
      } else {
        // All other cases - stand up
        shouldBeCrouched = false;
      }

      // Landing detection
      if (!wasGrounded.current && grounded) {
        jumpPhase.current = "land";
        setAnimation("jumpLand");
        if (footstepCooldownRef.current <= 0.05) {
          playFootstepSound();
          footstepCooldownRef.current = 0.25;
          const landingHits: Array<
            FootstepParticleSpawnOptions & { hitToi?: number }
          > = [];
          if (leftFootBone.current) {
            const pos = leftFootWorldPosition.current;
            leftFootBone.current.getWorldPosition(pos);
            const hit = castFootRay(pos);
            if (hit) landingHits.push(hit);
          }
          if (rightFootBone.current) {
            const pos = rightFootWorldPosition.current;
            rightFootBone.current.getWorldPosition(pos);
            const hit = castFootRay(pos);
            if (hit) landingHits.push(hit);
          }
          if (landingHits.length === 0 && character.current) {
            const centerPos = tempLandingCenterRef.current;
            character.current.getWorldPosition(centerPos);
            const centerHit = castFootRay(centerPos);
            if (centerHit) landingHits.push(centerHit);
          }
          if (landingHits.length === 0 && rb.current) {
            const fallbackPos = tempLandingCenterRef.current;
            const translation = rb.current.translation();
            const halfHeight = isCrouchingRef.current
              ? (capsuleHeight * 0.5) / 2
              : capsuleHeight / 2;
            fallbackPos.set(
              translation.x,
              translation.y - halfHeight - capsuleRadius,
              translation.z
            );
            const fallbackNormal = tempLandingNormalRef.current.set(0, 1, 0);
            landingHits.push({
              position: fallbackPos.clone(),
              normal: fallbackNormal.clone(),
              hitToi: 0,
            });
          }
          if (enableFootstepParticles) {
            landingHits.forEach((hit) => {
              footstepParticlesRef.current?.spawn(hit);
            });
          }
        }
        setTimeout(() => {
          if (jumpPhase.current === "land") {
            jumpPhase.current = "none";
          }
        }, 300);
      }

      // Handle roll input (Alt key) - only when grounded, not crouched, not dancing/attacking
      if (
        rollInput &&
        !rollPressed.current &&
        grounded &&
        !shouldBeCrouched &&
        !danceInput &&
        !isAttacking.current &&
        !isRolling.current
      ) {
        rollPressed.current = true;
        isRolling.current = true;
        jumpPhase.current = "none";
        setAnimation("roll");

        // Give a slight forward impulse in the current facing direction
        const rollSpeed = RUN_SPEED * 1.2;
        const facingRotation =
          rotationTarget.current + characterRotationTarget.current;
        vel.x = Math.sin(facingRotation) * rollSpeed;
        vel.z = Math.cos(facingRotation) * rollSpeed;

        setTimeout(() => {
          isRolling.current = false;
        }, 800);
      } else if (!rollInput) {
        rollPressed.current = false;
      }

      // If in air and not in jump phase, set to loop (unless crouching or transitioning)
      if (
        !grounded &&
        jumpPhase.current === "none" &&
        !shouldBeCrouched &&
        !crouchTransitioningRef.current &&
        !isRolling.current
      ) {
        jumpPhase.current = "loop";
        setAnimation("jumpLoop");
      }

      wasGrounded.current = grounded;

      const movement: any = { x: 0, z: 0 };

      // Handle dance input
      if (danceInput) {
        setAnimation("dance");
        movement.x = 0;
        movement.z = 0;
      }

      // Update capsule size when crouch state changes

      if (shouldBeCrouched !== isCrouchingRef.current) {
        const currentPos = rb.current.translation();
        const standingHalfHeight = capsuleHeight / 2;
        const crouchHalfHeight = (capsuleHeight * 0.5) / 2;
        const heightDiff = standingHalfHeight - crouchHalfHeight;

        // Set transitioning flag to prevent jump animations
        crouchTransitioningRef.current = true;

        // Reset jump phase to prevent fall animation
        jumpPhase.current = "none";

        setTimeout(() => {
          crouchTransitioningRef.current = false;
        }, 200); // 200ms grace period

        if (shouldBeCrouched) {
          // Crouching: move body DOWN
          rb.current.setTranslation(
            { x: currentPos.x, y: currentPos.y - heightDiff, z: currentPos.z },
            true
          );
        } else {
          // Standing up: move body UP (only if clearance!)
          rb.current.setTranslation(
            { x: currentPos.x, y: currentPos.y + heightDiff, z: currentPos.z },
            true
          );

          // Force idle animation when standing up
          setAnimation("idle");
        }

        // Update ref immediately (not async)
        isCrouchingRef.current = shouldBeCrouched;
      }

      // Movement input FIRST (before jump)
      if (get().forward) movement.z = 1;
      if (get().backward) movement.z = -1;
      if (get().left) movement.x = 1;
      if (get().right) movement.x = -1;

      // Q key: Classic walk backward
      const walkBackwardInput = get().walkBackward;
      if (walkBackwardInput) {
        movement.z = -1;
        movement.walkBackwardMode = true;
      }

      if (movement.x !== 0) {
        rotationTarget.current += ROTATION_SPEED * movement.x;
      }

      // Adjust speed based on run/crouch (use actual crouch state, not just input)
      let speed = get().run ? RUN_SPEED : WALK_SPEED;
      if (shouldBeCrouched) {
        speed = WALK_SPEED * 0.5; // Crouch walk is slower
      }

      if (movement.x !== 0 || movement.z !== 0) {
        // Calculate base movement direction from input
        const baseMovementAngle = movement.walkBackwardMode
          ? Math.atan2(movement.x, 1)
          : Math.atan2(movement.x, movement.z);

        // Movement direction is based on character rotation, NOT camera orbit
        // Camera orbit should only affect camera position, not movement direction
        const movementRotation = rotationTarget.current + baseMovementAngle;

        let intendedVelX = Math.sin(movementRotation) * speed;
        let intendedVelZ = Math.cos(movementRotation) * speed;

        // Character rotation should ONLY be based on movement input, NOT camera orbit
        // Camera orbit affects movement direction in world space, but character faces input direction
        // Character rotation is ALWAYS the same regardless of camera mode
        if (movement.walkBackwardMode) {
          characterRotationTarget.current = Math.atan2(movement.x, 1);
        } else {
          characterRotationTarget.current = Math.atan2(movement.x, movement.z);
        }

        if (movement.walkBackwardMode && movement.z < 0) {
          intendedVelX = -intendedVelX;
          intendedVelZ = -intendedVelZ;
        }

        // Apply velocity only when grounded OR when initiating jump
        if (grounded) {
          vel.x = intendedVelX;
          vel.z = intendedVelZ;
        }
        // When not grounded, don't touch velocity - let Rapier handle it

        // JUMP: Check if jumping this frame - apply horizontal momentum
        const jumpInput = get().jump;
        if (jumpInput && grounded && !jumpPressed.current) {
          jumpPressed.current = true;
          vel.y = JUMP_FORCE;
          // Apply horizontal velocity for forward jumping!
          vel.x = intendedVelX;
          vel.z = intendedVelZ;

          jumpPhase.current = "start";
          setAnimation("jumpStart");

          setTimeout(() => {
            if (jumpPhase.current === "start") {
              jumpPhase.current = "loop";
              setAnimation("jumpLoop");
            }
          }, 200);
        } else if (!jumpInput) {
          jumpPressed.current = false;
        }

        if (
          grounded &&
          jumpPhase.current === "none" &&
          !danceInput &&
          !isAttacking.current &&
          !isRolling.current
        ) {
          if (shouldBeCrouched) {
            setAnimation("crouchWalk");
          } else if (combatMode) {
            // Combat mode - use sword idle while moving (or could add sword walk)
            setAnimation("swordIdle");
          } else if (speed === RUN_SPEED) {
            setAnimation("run");
          } else if (movement.walkBackwardMode) {
            setAnimation("walkBackwards");
          } else {
            setAnimation("walk");
          }
        }
      } else {
        // No movement input

        if (grounded) {
          vel.x *= 0.85;
          vel.z *= 0.85;

          if (Math.abs(vel.x) < 0.01) vel.x = 0;
          if (Math.abs(vel.z) < 0.01) vel.z = 0;
        }

        // JUMP: Handle jumping when standing still
        const jumpInput = get().jump;
        if (jumpInput && grounded && !jumpPressed.current) {
          jumpPressed.current = true;
          vel.y = JUMP_FORCE;
          // No horizontal velocity - jump straight up

          jumpPhase.current = "start";
          setAnimation("jumpStart");

          setTimeout(() => {
            if (jumpPhase.current === "start") {
              jumpPhase.current = "loop";
              setAnimation("jumpLoop");
            }
          }, 200);
        } else if (!jumpInput) {
          jumpPressed.current = false;
        }

        if (
          grounded &&
          jumpPhase.current === "none" &&
          !danceInput &&
          !isAttacking.current &&
          !isRolling.current
        ) {
          if (shouldBeCrouched) {
            setAnimation("crouchIdle");
          } else if (combatMode) {
            setAnimation("swordIdle");
          } else {
            setAnimation("idle");
          }
        }
      }

      if (character.current) {
        // Character rotation should be independent of camera orbit
        // In follow-orbit mode, we need to subtract the camera orbit offset
        // so the character stays in place when camera orbits
        let targetRotation = characterRotationTarget.current;
        if (cameraMode === "follow-orbit") {
          // Counter-rotate to cancel out camera orbit effect
          // Character rotation should be relative to base camera, not orbiting camera
          targetRotation =
            characterRotationTarget.current - mouseOrbitOffset.current;
        }

        character.current.rotation.y = lerpAngle(
          character.current.rotation.y,
          targetRotation,
          0.1
        );
      }

      // Notify rotation change for third-person camera
      if (onRotationChange) {
        onRotationChange(
          rotationTarget.current + characterRotationTarget.current
        );
      }

      if (character.current) {
        character.current.updateMatrixWorld(true);
      }

      const horizontalSpeed = Math.hypot(vel.x, vel.z);
      const allowFootstepChecks =
        footstepAnimations.has(animationRef.current) &&
        horizontalSpeed > 0.2 &&
        !danceInput &&
        !isRolling.current &&
        !isAttacking.current;

      const processFoot = (
        boneRef: React.MutableRefObject<THREE.Object3D | null>,
        worldPosRef: React.MutableRefObject<THREE.Vector3>,
        prevPosRef: React.MutableRefObject<THREE.Vector3>,
        initializedRef: React.MutableRefObject<boolean>,
        wasGroundedRef: React.MutableRefObject<boolean>,
        prevToiRef: React.MutableRefObject<number>
      ): (FootstepParticleSpawnOptions & { hitToi?: number }) | null => {
        if (!boneRef.current) {
          return null;
        }

        boneRef.current.updateWorldMatrix(true, false);
        boneRef.current.getWorldPosition(worldPosRef.current);

        if (!initializedRef.current) {
          prevPosRef.current.copy(worldPosRef.current);
          initializedRef.current = true;
        }

        const verticalVelocity =
          (worldPosRef.current.y - prevPosRef.current.y) /
          Math.max(delta, 1e-4);
        const verticalDelta = prevPosRef.current.y - worldPosRef.current.y;
        const movementDelta = Math.sqrt(
          Math.pow(worldPosRef.current.x - prevPosRef.current.x, 2) +
            Math.pow(worldPosRef.current.z - prevPosRef.current.z, 2)
        );

        const hit = castFootRay(worldPosRef.current);
        const hitToi = hit?.hitToi ?? null;
        const slopeFactor = hit?.slopeFactor ?? 0;

        const groundedFoot =
          typeof hitToi === "number" &&
          hitToi < (slopeFactor > 0.4 ? 0.28 : 0.23);

        const triggered =
          allowFootstepChecks &&
          groundedFoot &&
          footstepCooldownRef.current <= 0 &&
          ((!wasGroundedRef.current &&
            (verticalVelocity < -0.02 ||
              verticalDelta > 0.006 ||
              slopeFactor > 0.35)) ||
            (typeof hitToi === "number" &&
              prevToiRef.current - hitToi >
                (slopeFactor > 0.2 ? 0.025 : 0.05) &&
              movementDelta > 0.01));

        prevPosRef.current.copy(worldPosRef.current);
        wasGroundedRef.current = groundedFoot;
        prevToiRef.current =
          groundedFoot && typeof hitToi === "number" ? hitToi : 1;

        return triggered ? hit : null;
      };

      const leftHit = processFoot(
        leftFootBone,
        leftFootWorldPosition,
        prevLeftFootPosition,
        leftFootInitialized,
        leftFootWasGrounded,
        leftFootPrevToi
      );
      const rightHit = processFoot(
        rightFootBone,
        rightFootWorldPosition,
        prevRightFootPosition,
        rightFootInitialized,
        rightFootWasGrounded,
        rightFootPrevToi
      );

      const hitsToProcess: Array<
        FootstepParticleSpawnOptions & { hitToi?: number }
      > = [];
      if (leftHit) {
        hitsToProcess.push(leftHit);
      }
      if (rightHit) {
        hitsToProcess.push(rightHit);
      }

      if (hitsToProcess.length > 0) {
        if (enableFootstepAudio) {
          playFootstepSound();
        }
        footstepCooldownRef.current = 0.2;
        if (enableFootstepParticles) {
          hitsToProcess.forEach((hit) => {
            footstepParticlesRef.current?.spawn(hit);
          });
        }
      }

      rb.current.setLinvel(vel, true);
    }

    // CAMERA
    if (cameraMode === "follow" || cameraMode === "follow-orbit") {
      // Base rotation follows character movement
      const baseRotation = rotationTarget.current;

      // For follow-orbit, add mouse orbit offset to camera rotation ONLY
      // The container holds camera position, so camera can orbit without affecting character
      const finalRotation =
        cameraMode === "follow-orbit"
          ? baseRotation + mouseOrbitOffset.current
          : baseRotation;

      container.current.rotation.y = MathUtils.lerp(
        container.current.rotation.y,
        finalRotation,
        cameraLerpSpeed
      );

      // IMPORTANT: character rotation is NOT affected by camera orbit
      // character.current.rotation.y is set separately based on movement only

      cameraPosition.current.getWorldPosition(cameraWorldPosition.current);

      const isFirstFrame = !cameraInitialized.current;

      if (isFirstFrame) {
        camera.position.copy(cameraWorldPosition.current);
      } else {
        camera.position.lerp(cameraWorldPosition.current, cameraLerpSpeed);
      }

      if (cameraTarget.current) {
        cameraTarget.current.getWorldPosition(
          cameraLookAtWorldPosition.current
        );

        if (isFirstFrame) {
          cameraLookAt.current.copy(cameraLookAtWorldPosition.current);
        } else {
          cameraLookAt.current.lerp(
            cameraLookAtWorldPosition.current,
            cameraLerpSpeed
          );
        }

        // For follow-orbit, adjust camera look-at target vertically based on accumulated vertical mouse offset
        if (cameraMode === "follow-orbit") {
          // Apply vertical rotation offset (pitch) by adjusting look-at target height
          // mouseVerticalOffset is already accumulated and clamped, convert to vertical offset
          const verticalRotationOffset =
            Math.sin(mouseVerticalOffset.current) * 2;
          cameraLookAt.current.y =
            cameraLookAtWorldPosition.current.y + verticalRotationOffset;
        }

        camera.lookAt(cameraLookAt.current);
      }

      if (isFirstFrame) {
        cameraInitialized.current = true;
      }
    }

    if (teleportHoldFramesRef.current > 0) {
      teleportHoldFramesRef.current -= 1;
      camera.position.copy(teleportCameraPositionRef.current);
      smoothCameraPosition.current.copy(teleportCameraPositionRef.current);
      cameraLookAt.current.copy(teleportLookAtRef.current);
      smoothLookAtPosition.current.copy(teleportLookAtRef.current);
      camera.lookAt(teleportLookAtRef.current);
    }
  });

  return (
    <>
      <RigidBody
        colliders={false}
        ref={rb}
        position={position}
        gravityScale={1}
        enabledRotations={[false, false, false]}
        type="dynamic"
        ccd={true}
      >
        <group ref={container}>
          <group ref={cameraTarget} position-z={targetZ} />
          <group ref={cameraPosition} position={[cameraX, cameraY, cameraZ]} />
          <group
            ref={character}
            position-y={
              isCrouchingRef.current
                ? capsuleHeight / 2 - (capsuleHeight * 0.5) / 2
                : 0
            }
          >
            <GodotCharacter
              animation={animation}
              onFootBonesReady={handleFootBonesReady}
            />
          </group>
        </group>
        <CapsuleCollider
          args={[
            isCrouchingRef.current
              ? (capsuleHeight * 0.5) / 2
              : capsuleHeight / 2,
            capsuleRadius,
          ]}
          friction={0.5}
          restitution={0}
        />
      </RigidBody>
      {enableFootstepParticles && (
        <FootstepParticles ref={footstepParticlesRef} />
      )}
    </>
  );
};
