from pathlib import Path

import numpy as np
from moviepy import (
    ColorClip,
    CompositeVideoClip,
    ImageClip,
    TextClip,
    VideoFileClip,
    clips_array,
    concatenate_videoclips,
)


def _with_bottom_index(clip: VideoFileClip, index: int, label_height: int = 42) -> CompositeVideoClip:
    width, height = clip.size
    label_bg = ColorClip(size=(width, label_height), color=(20, 20, 20), duration=clip.duration)
    label_text = TextClip(
        text=str(index),
        font_size=int(label_height * 0.7),
        color="white",
        method="caption",
        size=(width, label_height),
    ).with_duration(clip.duration)
    canvas = CompositeVideoClip(
        [
            clip.with_position((0, 0)),
            label_bg.with_position((0, height)),
            label_text.with_position((0, height)),
        ],
        size=(width, height + label_height),
    )
    return canvas.with_duration(clip.duration)


def _pad_clip_with_last_frame(clip: VideoFileClip, target_duration: float) -> VideoFileClip:
    if clip.duration >= target_duration:
        return clip.subclipped(0, target_duration)

    last_t = max(clip.duration - 1e-3, 0)
    last_frame = clip.get_frame(last_t)
    frozen_tail = ImageClip(last_frame).with_duration(target_duration - clip.duration)
    return concatenate_videoclips([clip, frozen_tail], method="compose")


def _auto_crop_vertical_whitespace(clip: VideoFileClip) -> VideoFileClip:
    frame = clip.get_frame(0)
    if frame.ndim != 3 or frame.shape[0] < 10:
        return clip

    row_mean = frame.mean(axis=(1, 2))
    row_std = frame.std(axis=(1, 2))
    is_blank = (row_mean > 245) & (row_std < 12)

    top = 0
    while top < len(is_blank) and is_blank[top]:
        top += 1

    bottom = 0
    while bottom < len(is_blank) and is_blank[len(is_blank) - 1 - bottom]:
        bottom += 1

    max_crop = int(frame.shape[0] * 0.25)
    top = min(top, max_crop)
    bottom = min(bottom, max_crop)

    if top == 0 and bottom == 0:
        return clip

    new_y1 = top
    new_y2 = frame.shape[0] - bottom
    if new_y2 - new_y1 < int(frame.shape[0] * 0.5):
        return clip

    return clip.cropped(y1=new_y1, y2=new_y2)


def build_gif_group(
    group_files: list[Path],
    output_path: Path,
    start_index: int,
    target_width: int = 420,
    fps: int = 12,
) -> None:
    clips = []
    for path in group_files:
        try:
            clip = VideoFileClip(str(path))
        except Exception as exc:
            print(f"跳过损坏视频: {path.name} ({exc})")
            continue
        clips.append(clip)

    if not clips:
        print(f"跳过输出 {output_path.name}：该组没有可用视频")
        return

    try:
        max_duration = max(clip.duration for clip in clips)
        prepared = []
        for offset, clip in enumerate(clips):
            adjusted = clip
            if start_index == 1:
                adjusted = _auto_crop_vertical_whitespace(adjusted)
            resized = adjusted.resized(width=target_width)
            padded = _pad_clip_with_last_frame(resized, max_duration)
            prepared.append(_with_bottom_index(padded, start_index + offset))
        if start_index == 1:
            final_clip = clips_array([[clip] for clip in prepared])
        else:
            final_clip = clips_array([prepared])
        final_clip.write_gif(str(output_path), fps=fps)
        final_clip.close()
        for clip in prepared:
            clip.close()
    finally:
        for clip in clips:
            clip.close()


def main() -> None:
    work_dir = Path(__file__).resolve().parent
    mp4_files = sorted(work_dir.glob("*.mp4"))

    if len(mp4_files) < 1:
        raise ValueError("未找到 mp4 文件")

    if len(mp4_files) != 9:
        print(f"警告：期望 9 个 mp4，当前找到 {len(mp4_files)} 个，将按现有文件尽量生成")

    groups = [mp4_files[i : i + 3] for i in range(0, len(mp4_files), 3)]

    for index, group in enumerate(groups, start=1):
        output_name = work_dir / f"combined_row_{index}.gif"
        print(f"正在生成: {output_name.name}")
        build_gif_group(group, output_name, start_index=(index - 1) * 3 + 1)

    print("完成：已生成 3 个横排 GIF。")


if __name__ == "__main__":
    main()