import json
import numpy as np
import pyvista as pv
import random as rd
import argparse
import os
from typing import Tuple, List
from PIL import Image

MIN_R = 2.0
MAX_R = 6.0
MAX_DEG_PER_STEP = 90.0

SHAPE_REGISTRY = {
    "tetrahedron": 0,
    "cube": 1,
    "octahedron": 2,
    "dodecahedron": 3,
    "icosahedron": 4,
}
SHAPE_NAMES = list(SHAPE_REGISTRY.keys())


def get_mesh(kind: str) -> pv.PolyData:
    return pv.PlatonicSolid(kind, radius=0.4, center=(0, 0, 0))


def closed_loop_trajectory(
    length: int, initial_state: Tuple[float, float, float]
) -> List[Tuple[float, float, float]]:

    velocities = []

    sd_0, sy_0, sx_0 = initial_state
    sd_i, sy_i, sx_i = initial_state
    vd_i, vy_i, vx_i = 0.0, 0.0, 0.0

    if length % 2 != 0:
        length += 1

    for i in range(length):
        if i == length - 1:
            vd_i = -(sd_i - sd_0)
            vy_i = -(sy_i - sy_0)
            vx_i = -(sx_i - sx_0)
        else:
            vd_i = rd.uniform(-1 - sd_i, 1 - sd_i)
            vx_i = rd.uniform(-1, 1)
            vy_i = rd.uniform(-1, 1)

            sd_i += vd_i
            sx_i += vx_i
            sy_i += vy_i

        velocities.append((vd_i, vy_i, vx_i))

    return velocities


def repeated_vel_closed_loop_trajectory(
    length: int, initial_state: Tuple[float, float, float]
) -> List[Tuple[float, float, float]]:
    velocities = []

    sd_0, sy_0, sx_0 = initial_state
    sd_i, sy_i, sx_i = initial_state
    vd_i, vy_i, vx_i = 0.0, 0.0, 0.0

    if length % 2 != 0:
        length += 1

    for i in range(length):
        if i == length - 1 or i == length - 2:
            vd_i = -(sd_i - sd_0) / 2
            vy_i = -(sy_i - sy_0) / 2
            vx_i = -(sx_i - sx_0) / 2
        else:
            if i % 2 == 0:
                vd_i = rd.uniform(-1 - sd_i, 1 - sd_i) / 2
                vx_i = rd.uniform(-1, 1) / 2
                vy_i = rd.uniform(-1, 1) / 2

            sd_i += vd_i
            sx_i += vx_i
            sy_i += vy_i

        velocities.append((vd_i, vy_i, vx_i))

    return velocities


def apply_action_and_get_image(
    plotter: pv.Plotter,
    actor: pv.Actor,
    current_state: Tuple[float, float, float],
    action_vel: Tuple[float, float, float],
) -> Tuple[np.ndarray, Tuple[float, float, float]]:

    d0, ry0, rx0 = current_state
    vd, vy, vx = action_vel

    d_final = np.clip(d0 + vd, -1, 1)
    ry_final = ry0 + (vy * MAX_DEG_PER_STEP)
    rx_final = rx0 + (vx * MAX_DEG_PER_STEP)

    r = ((MAX_R + MIN_R) / 2) + (d_final * (MAX_R - MIN_R) / 2)

    plotter.camera.position = (0, 0, r)
    plotter.camera.focal_point = (0, 0, 0)
    plotter.camera.up = (0, 1, 0)

    actor.orientation = [rx_final, ry_final, 0]

    plotter.render()
    img = plotter.screenshot(transparent_background=False, return_img=True)

    return img, (d_final, ry_final, rx_final)


def generate_raw_dataset(
    output_dir: str,
    num_trajectories: int,
    trajectory_length: int,
    resolution: int,
    shape_arg: str,
    monochromatic: bool = False,
    repeated_vel: bool = True,
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pl = pv.Plotter(window_size=[resolution, resolution], off_screen=True)
    pl.set_background("white")
    pl.hide_axes()

    print(f"Generating {num_trajectories} trajectories ({shape_arg}).")

    if shape_arg == "mixed":
        assigned_shapes = [
            SHAPE_NAMES[i % len(SHAPE_NAMES)] for i in range(num_trajectories)
        ]
        rd.shuffle(assigned_shapes)
    else:
        assigned_shapes = [shape_arg] * num_trajectories

    for i in range(num_trajectories):
        traj_imgs = []
        traj_vels = []
        traj_states = []

        current_shape_name = assigned_shapes[i]
        current_shape_id = SHAPE_REGISTRY[current_shape_name]

        pl.clear_actors()
        mesh = get_mesh(current_shape_name)

        if monochromatic:
            actor = pl.add_mesh(mesh, color="white", show_edges=True, line_width=2)
        else:
            if "face_ids" not in mesh.cell_data:
                mesh.cell_data["face_ids"] = np.arange(mesh.n_cells)

            actor = pl.add_mesh(
                mesh,
                show_edges=True,
                line_width=2,
                cmap="tab20",
                scalars="face_ids",
                preference="cell",
                show_scalar_bar=False,
            )

        start_d = rd.uniform(-1, 1)
        start_ry = rd.uniform(0, 360)
        start_rx = rd.uniform(0, 360)
        state = (start_d, start_ry, start_rx)

        img, state = apply_action_and_get_image(pl, actor, state, (0.0, 0.0, 0.0))
        traj_imgs.append(img)
        traj_states.append(state)

        if repeated_vel:
            velocities = repeated_vel_closed_loop_trajectory(trajectory_length, state)
        else:
            velocities = closed_loop_trajectory(trajectory_length, state)

        for vel in velocities:
            img, state = apply_action_and_get_image(pl, actor, state, vel)
            traj_imgs.append(img)
            traj_states.append(state)
            traj_vels.append(vel)

        # I/O Flush to Hardware
        base_name = f"traj_{i:06d}"
        
        for f_idx, frame_arr in enumerate(traj_imgs):
            img_path = os.path.join(output_dir, f"{base_name}.frame_{f_idx:03d}.jpg")
            Image.fromarray(frame_arr).save(img_path, quality=95)

        np.save(os.path.join(output_dir, f"{base_name}.actions.npy"), np.array(traj_vels, dtype=np.float32))
        np.save(os.path.join(output_dir, f"{base_name}.states.npy"), np.array(traj_states, dtype=np.float32))
        
        with open(os.path.join(output_dir, f"{base_name}.meta.json"), "w") as f:
            json.dump({"shape_id": current_shape_id, "shape_name": current_shape_name}, f)

        print(f"  [I/O] Flushed {base_name} to disk | Progress: {i+1}/{num_trajectories}", end="\r")

    print("\n")
    pl.close()
    print(f"Done. {num_trajectories} trajectories generated as raw media.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_trajs", type=int, default=100)
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--output_dir", type=str, default="../sample_trajectories/train")
    parser.add_argument("--monochromatic", action="store_true")
    parser.add_argument(
        "--shape", type=str, default="icosahedron", choices=SHAPE_NAMES + ["mixed"]
    )
    parser.add_argument("--repeated_vel", type=bool, default=False)

    args = parser.parse_args()

    generate_raw_dataset(
        args.output_dir,
        args.num_trajs,
        args.length,
        args.resolution,
        args.shape,
        args.monochromatic,
        args.repeated_vel,
    )