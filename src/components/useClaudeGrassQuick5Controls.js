import { useControls, folder } from "leva";

export default function useClaudeGrassQuick5Controls() {
  const controls = useControls("🌿 FOLIAGE", {
    claudeGrassQuick5: folder(
      {
        // Master toggle
        enabled: {
          value: false,
          label: "🌿 Enable Claude Grass Quick 5",
        },

        // Basic grass parameters
        grassHeight: {
          value: 0.8,
          min: 0.1,
          max: 5,
          step: 0.1,
          label: "📏 Grass Height",
        },
        grassWidth: {
          value: 0.1,
          min: 0.01,
          max: 1,
          step: 0.01,
          label: "📐 Grass Width",
        },
        grassDensity: {
          value: 3072,
          min: 256,
          max: 10000,
          step: 256,
          label: "🌱 Grass Density (Blades per Patch)",
        },

        // Grid and spacing
        gridSize: {
          value: 16,
          min: 4,
          max: 32,
          step: 1,
          label: "📐 Grid Size (Render Distance)",
        },
        patchSpacing: {
          value: 10,
          min: 5,
          max: 30,
          step: 1,
          label: "📏 Patch Spacing",
        },
        patchSize: {
          value: 10,
          min: 5,
          max: 20,
          step: 1,
          label: "📦 Patch Size",
        },

        // LOD settings
        lodDistance: {
          value: 15,
          min: 5,
          max: 50,
          step: 1,
          label: "👁️ LOD Distance",
        },
        maxDistance: {
          value: 100,
          min: 50,
          max: 200,
          step: 10,
          label: "🚀 Max Distance",
        },

        // Terrain
        terrainSize: {
          value: 100,
          min: 50,
          max: 500,
          step: 10,
          label: "🗺️ Terrain Size",
        },
        heightScale: {
          value: 1,
          min: 0,
          max: 10,
          step: 0.1,
          label: "⬆️ Height Scale",
        },
        heightOffset: {
          value: 0,
          min: -10,
          max: 10,
          step: 0.1,
          label: "↕️ Height Offset",
        },

        // Grass Colors
        colors: folder(
          {
            baseColor1: {
              value: "#051303",
              label: "Base Color 1 (Dark)",
            },
            baseColor2: {
              value: "#061a03",
              label: "Base Color 2 (Light)",
            },
            tipColor1: {
              value: "#a6cc40",
              label: "Tip Color 1 (Dark)",
            },
            tipColor2: {
              value: "#cce666",
              label: "Tip Color 2 (Light)",
            },
            gradientCurve: {
              value: 4.0,
              min: 0.5,
              max: 10.0,
              step: 0.1,
              label: "📈 Gradient Curve (Steepness)",
            },
          },
          { collapsed: true }
        ),

        // Wind Controls
        wind: folder(
          {
            windEnabled: {
              value: true,
              label: "Enable Wind",
            },
            windStrength: {
              value: 1.25,
              min: 0.0,
              max: 3.0,
              step: 0.05,
              label: "Wind Strength",
            },
            windDirectionScale: {
              value: 0.05,
              min: 0.01,
              max: 0.2,
              step: 0.01,
              label: "Wind Direction Scale",
            },
            windDirectionSpeed: {
              value: 0.05,
              min: 0.01,
              max: 0.5,
              step: 0.01,
              label: "Wind Direction Speed",
            },
            windStrengthScale: {
              value: 0.25,
              min: 0.1,
              max: 1.0,
              step: 0.05,
              label: "Wind Strength Scale",
            },
            windStrengthSpeed: {
              value: 1.0,
              min: 0.1,
              max: 5.0,
              step: 0.1,
              label: "Wind Strength Speed",
            },
          },
          { collapsed: true }
        ),

        // Advanced Parameters
        advanced: folder(
          {
            aoEnabled: {
              value: true,
              label: "Enable AO",
            },
            aoIntensity: {
              value: 1.0,
              min: 0,
              max: 2,
              step: 0.1,
              label: "AO Intensity",
            },
            grassMiddleBrightnessMin: {
              value: 0.85,
              min: 0.0,
              max: 1.0,
              step: 0.01,
              label: "Middle Brightness (Min)",
            },
            grassMiddleBrightnessMax: {
              value: 1.0,
              min: 0.0,
              max: 1.0,
              step: 0.01,
              label: "Edge Brightness (Max)",
            },
          },
          { collapsed: true }
        ),

        // Specular Controls
        specular: folder(
          {
            specularEnabled: {
              value: false,
              label: "Enable Specular",
            },
            specularIntensity: {
              value: 2.0,
              min: 0.0,
              max: 5.0,
              step: 0.1,
              label: "Intensity",
            },
            specularColor: {
              value: "#ffffff",
              label: "Color",
            },
            specularDirectionX: {
              value: -1.0,
              min: -1.0,
              max: 1.0,
              step: 0.1,
              label: "Direction X",
            },
            specularDirectionY: {
              value: 1.0,
              min: -1.0,
              max: 1.0,
              step: 0.1,
              label: "Direction Y",
            },
            specularDirectionZ: {
              value: 0.5,
              min: -1.0,
              max: 1.0,
              step: 0.1,
              label: "Direction Z",
            },
          },
          { collapsed: true }
        ),

        // Backscatter/SSS Controls
        backscatter: folder(
          {
            backscatterEnabled: {
              value: true,
              label: "Enable Backscatter",
            },
            backscatterIntensity: {
              value: 0.5,
              min: 0,
              max: 2,
              step: 0.1,
              label: "Backscatter Intensity",
            },
            backscatterColor: {
              value: "#51cc66",
              label: "Backscatter Color",
            },
            backscatterPower: {
              value: 2.0,
              min: 0.5,
              max: 5,
              step: 0.1,
              label: "Backscatter Power",
            },
            frontScatterStrength: {
              value: 0.3,
              min: 0,
              max: 1,
              step: 0.1,
              label: "Front Scatter",
            },
            rimSSSStrength: {
              value: 0.5,
              min: 0,
              max: 1,
              step: 0.1,
              label: "Rim SSS Strength",
            },
          },
          { collapsed: true }
        ),

        // Fog Controls
        fog: folder(
          {
            fogEnabled: {
              value: false,
              label: "Enable Fog",
            },
            fogNear: {
              value: 5.0,
              min: 0,
              max: 100,
              step: 0.5,
              label: "Fog Start",
            },
            fogFar: {
              value: 50.0,
              min: 0,
              max: 200,
              step: 1,
              label: "Fog End",
            },
            fogIntensity: {
              value: 1.0,
              min: 0,
              max: 2,
              step: 0.1,
              label: "Fog Intensity",
            },
            fogColor: {
              value: "#4f74af",
              label: "Fog Color",
            },
          },
          { collapsed: true }
        ),

        // Player Interaction Controls
        playerInteraction: folder(
          {
            playerInteractionEnabled: {
              value: true,
              label: "Enable Player Interaction",
            },
            playerInteractionRepel: {
              value: true,
              label: "Repel (off = Attract)",
            },
            playerInteractionRange: {
              value: 2.5,
              min: 0.5,
              max: 10.0,
              step: 0.1,
              label: "Interaction Range",
            },
            playerInteractionStrength: {
              value: 0.2,
              min: 0.0,
              max: 1.0,
              step: 0.05,
              label: "Interaction Strength",
            },
            playerInteractionHeightThreshold: {
              value: 3.0,
              min: 1.0,
              max: 10.0,
              step: 0.1,
              label: "Height Threshold",
            },
          },
          { collapsed: true }
        ),
      },
      { collapsed: true }
    ),
  });

  return controls;
}
