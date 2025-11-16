import {
  Environment,
  OrbitControls,
  OrthographicCamera,
  Sky,
} from "@react-three/drei";
import { Physics } from "@react-three/rapier";
import { useControls } from "leva";
import { useEffect, useRef, useState } from "react";
import { CharacterController } from "./CharacterController";
import { HeightFog } from "./HeightFog";
import { HillySideScrollerMap } from "./HillySideScrollerMap";
import { Map } from "./Map";
import { ProceduralTerrain } from "./ProceduralTerrain";
import { SideScrollerMap } from "./SideScrollerMap";
import useClaudeGrassQuick3Controls from "./useClaudeGrassQuick3Controls";

const maps = {
  castle_on_hills: {
    scale: 3,
    position: [-6, -7, 0],
    type: "glb",
  },
  animal_crossing_map: {
    scale: 20,
    position: [-15, -1, 10],
    type: "glb",
  },
  city_scene_tokyo: {
    scale: 0.72,
    position: [0, -1, -3.5],
    type: "glb",
  },
  de_dust_2_with_real_light: {
    scale: 0.3,
    position: [-5, -3, 13],
    type: "glb",
  },
  medieval_fantasy_book: {
    scale: 0.4,
    position: [-4, 0, -6],
    type: "glb",
  },
  procedural_terrain: {
    scale: 1,
    position: [0, 0, 0],
    type: "procedural",
  },
  side_scroller_city: {
    scale: 1,
    position: [0, 0, 0],
    type: "sidescroller",
  },
  hilly_side_scroller: {
    scale: 1,
    position: [0, 0, 0],
    type: "hillysidescroller",
  },
};

export const Experience = () => {
  const shadowCameraRef = useRef();
  const [terrainReady, setTerrainReady] = useState(false);
  const playerPositionRef = useRef([0, 0, 0]);
  const [playerPosition, setPlayerPosition] = useState([0, 0, 0]);
  const { map } = useControls("Map", {
    map: {
      value: "castle_on_hills",
      options: Object.keys(maps),
    },
  });
  const { enableOrbitControls } = useControls("Camera", {
    enableOrbitControls: {
      value: false,
      label: "Enable Orbit Controls",
    },
  });
  const { showPhysicsDebug } = useControls("Debug", {
    showPhysicsDebug: {
      value: false,
      label: "Show Physics Debug",
    },
  });

  // Get grass controls
  const grassControls = useClaudeGrassQuick3Controls();

  // HeightFog controls
  const fogControls = useControls("Height Fog", {
    enabled: {
      value: true,
      label: "Enabled",
    },
    fogColor: {
      value: "#cccccc",
      label: "Fog Color",
    },
    fogHeight: {
      value: 200.0,
      min: 0,
      max: 200,
      step: 1,
      label: "Fog Height",
    },
    fogNear: {
      value: 1,
      min: 0,
      max: 100,
      step: 1,
      label: "Fog Near",
    },
    fogFar: {
      value: 200,
      min: 100,
      max: 5000,
      step: 50,
      label: "Fog Far",
    },
  });

  // Sky controls
  const skyControls = useControls("Sky", {
    enabled: {
      value: true,
      label: "Enabled",
    },
    distance: {
      value: 450000,
      min: 100000,
      max: 1000000,
      step: 10000,
      label: "Distance",
    },
    sunPositionX: {
      value: 100,
      min: -500,
      max: 500,
      step: 10,
      label: "Sun Position X",
    },
    sunPositionY: {
      value: 20,
      min: -100,
      max: 500,
      step: 5,
      label: "Sun Position Y",
    },
    sunPositionZ: {
      value: 100,
      min: -500,
      max: 500,
      step: 10,
      label: "Sun Position Z",
    },
    inclination: {
      value: 0.6,
      min: 0,
      max: 1,
      step: 0.01,
      label: "Inclination",
    },
    azimuth: {
      value: 0.25,
      min: 0,
      max: 1,
      step: 0.01,
      label: "Azimuth",
    },
    turbidity: {
      value: 8,
      min: 0,
      max: 20,
      step: 0.1,
      label: "Turbidity",
    },
    rayleigh: {
      value: 2,
      min: 0,
      max: 4,
      step: 0.1,
      label: "Rayleigh",
    },
    mieCoefficient: {
      value: 0.005,
      min: 0,
      max: 0.1,
      step: 0.001,
      label: "Mie Coefficient",
    },
    mieDirectionalG: {
      value: 0.8,
      min: 0,
      max: 1,
      step: 0.01,
      label: "Mie Directional G",
    },
  });

  // Delay character spawn to allow terrain to load
  useEffect(() => {
    setTerrainReady(false);
    const timer = setTimeout(() => {
      setTerrainReady(true);
    }, 1000); // 1 second delay

    return () => clearTimeout(timer);
  }, [map]);

  return (
    <>
      {enableOrbitControls && <OrbitControls />}
      <HeightFog {...fogControls} />
      {skyControls.enabled && (
        <Sky
          distance={skyControls.distance}
          sunPosition={[
            skyControls.sunPositionX,
            skyControls.sunPositionY,
            skyControls.sunPositionZ,
          ]}
          inclination={skyControls.inclination}
          azimuth={skyControls.azimuth}
          turbidity={skyControls.turbidity}
          rayleigh={skyControls.rayleigh}
          mieCoefficient={skyControls.mieCoefficient}
          mieDirectionalG={skyControls.mieDirectionalG}
        />
      )}
      <Environment preset="sunset" />
      <directionalLight
        intensity={0.65}
        castShadow
        position={[-15, 10, 15]}
        shadow-mapSize-width={2048}
        shadow-mapSize-height={2048}
        shadow-bias={-0.00005}
      >
        <OrthographicCamera
          left={-22}
          right={15}
          top={10}
          bottom={-20}
          ref={shadowCameraRef}
          attach={"shadow-camera"}
        />
      </directionalLight>
      <Physics
        key={map}
        debug={showPhysicsDebug}
        gravity={[0, -9.81, 0]}
        updateLoop="independent"
      >
        {maps[map].type === "procedural" ? (
          <ProceduralTerrain
            scale={maps[map].scale}
            position={maps[map].position}
          />
        ) : maps[map].type === "sidescroller" ? (
          <SideScrollerMap
            scale={maps[map].scale}
            position={maps[map].position}
            playerPosition={playerPosition}
            grassControls={grassControls}
          />
        ) : maps[map].type === "hillysidescroller" ? (
          <HillySideScrollerMap
            scale={maps[map].scale}
            position={maps[map].position}
            playerPosition={playerPosition}
            grassControls={grassControls}
          />
        ) : (
          <Map
            scale={maps[map].scale}
            position={maps[map].position}
            model={`models/${map}.glb`}
          />
        )}
        {terrainReady && (
          <CharacterController
            mapType={maps[map].type}
            enableOrbitControls={enableOrbitControls}
            playerPositionRef={playerPositionRef}
            setPlayerPosition={setPlayerPosition}
          />
        )}
      </Physics>
    </>
  );
};
