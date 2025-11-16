import { useControls, folder } from "leva";

export function useGFS2Controls() {
  return useControls("🌿 FOLIAGE", {
    gfs2: folder({
      gfs2Enabled: {
        value: false,
        label: "🌿 Enable GFS2 (Shader Grass)",
      },
      showNormals: {
        value: false,
        label: "🔍 Show Normals (Debug)",
      },
    }),
  });
}

