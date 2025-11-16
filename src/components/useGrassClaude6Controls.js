import { useControls, folder } from "leva";

export function useGrassClaude6Controls() {
  return useControls("🌿 FOLIAGE", {
    grassClaude6: folder({
      grassClaude6Enabled: {
        value: false,
        label: "🌿 Enable Grass Claude 6",
      },
    }),
  });
}
