import { useControls, folder } from "leva";

export function useGrassFromScratchControls() {
  return useControls("🌿 FOLIAGE", {
    grassFromScratch: folder({
      grassFromScratchEnabled: {
        value: false,
        label: "🌿 Enable Grass From Scratch",
      },
    }),
  });
}

