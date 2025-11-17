import { useControls } from "leva";

export function useGFS3Controls() {
  return useControls("GFS3", {
    gfs3Enabled: {
      value: false,
      label: "Enable GFS3",
    },
  });
}
