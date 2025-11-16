import { useControls, folder } from "leva";

export function useGrassClaude7Controls() {
  return useControls("🌿 FOLIAGE", {
    grassClaude7: folder({
      grassClaude7Enabled: {
        value: false,
        label: "🌿 Enable Grass Claude 7",
      },
    }),
  });
}
