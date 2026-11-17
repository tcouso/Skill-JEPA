"""Composes trajectory keyframes + goal frame into one PIL image for wandb logging."""

import numpy as np
from PIL import Image, ImageDraw

THUMB_SIZE = 112
LABEL_HEIGHT = 18
DIVIDER_HEIGHT = 6
GOAL_BORDER = 3


def make_contact_sheet(
    frames: list,
    goal_frame: np.ndarray | None,
    max_frames: int = 10,
    success: bool | None = None,
    total_steps: int | None = None,
) -> Image.Image:
    """Trajectory keyframes over `total_steps` on top, goal frame below, tinted by `success`."""

    def to_pil_thumb(arr: np.ndarray) -> Image.Image:
        while arr.ndim > 3:
            arr = arr[-1]  # strip batch/time dims, keep most recent frame
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
            arr = arr.transpose(1, 2, 0)  # CHW → HWC
        if arr.dtype != np.uint8:
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr).resize((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)

    total_width = max_frames * THUMB_SIZE
    total_height = THUMB_SIZE + LABEL_HEIGHT + DIVIDER_HEIGHT + THUMB_SIZE + LABEL_HEIGHT
    sheet = Image.new("RGB", (total_width, total_height), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)

    outcome_color = (
        (70, 70, 70) if success is None
        else (60, 170, 90) if success
        else (200, 60, 60)
    )

    n = len(frames)
    span = max(total_steps if total_steps else n, n)
    if n > 0:
        indices = np.linspace(0, span - 1, min(max_frames, span), dtype=int)
        in_range = np.nonzero(indices < n)[0]
        last_col = int(in_range[-1]) if in_range.size else None
        if last_col is not None:
            indices[last_col] = n - 1
        for col, idx in enumerate(indices):
            x = col * THUMB_SIZE
            if idx < n:
                sheet.paste(to_pil_thumb(frames[idx]), (x, 0))
                label_fill = (180, 180, 180)
                if col == last_col and success is not None:
                    draw.rectangle(
                        [x, 0, x + THUMB_SIZE - 1, THUMB_SIZE - 1],
                        outline=outcome_color,
                        width=3,
                    )
                    label_fill = outcome_color
            else:
                draw.rectangle(
                    [x + 1, 1, x + THUMB_SIZE - 2, THUMB_SIZE - 2], fill=(38, 38, 38)
                )
                draw.text((x + THUMB_SIZE // 2 - 14, THUMB_SIZE // 2 - 4), "ended",
                          fill=(110, 110, 110))
                label_fill = (110, 110, 110)
            draw.text((x + THUMB_SIZE // 2, THUMB_SIZE + 2), f"t={idx}", fill=label_fill)

    div_y = THUMB_SIZE + LABEL_HEIGHT
    draw.rectangle([0, div_y, total_width - 1, div_y + DIVIDER_HEIGHT - 1], fill=outcome_color)

    goal_y = div_y + DIVIDER_HEIGHT
    if goal_frame is not None:
        inner_size = THUMB_SIZE - 2 * GOAL_BORDER
        inner = to_pil_thumb(goal_frame).resize((inner_size, inner_size), Image.LANCZOS)
        bordered = Image.new("RGB", (THUMB_SIZE, THUMB_SIZE), (210, 50, 50))
        bordered.paste(inner, (GOAL_BORDER, GOAL_BORDER))
        goal_x = (total_width - THUMB_SIZE) // 2
        sheet.paste(bordered, (goal_x, goal_y))
        draw.text((total_width // 2 - 12, goal_y + THUMB_SIZE + 2), "GOAL", fill=(210, 50, 50))

    if success is not None:
        draw.text(
            (4, goal_y + THUMB_SIZE + 2),
            "SUCCESS" if success else "FAILURE",
            fill=outcome_color,
        )

    return sheet
