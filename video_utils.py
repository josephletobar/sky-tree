import cv2


def uniform_samples(start, end, n):
    n = min(n, end - start)
    if n <= 0:
        return []
    if n == 1:
        return [start]
    return [round(start + i * (end - start - 1) / (n - 1)) for i in range(n)]


def split_to_n(start, end, n):
    frames_per_segment = (end - start) // n
    return [(start + i * frames_per_segment,
             start + (i + 1) * frames_per_segment if i < n - 1 else end)
            for i in range(n)]


def sample_frames(cap, start, end, n):
    images = []
    for index in uniform_samples(start, end, n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f"Could not read frame {index}")
        ok, image = cv2.imencode(".jpg", frame)
        if not ok:
            raise ValueError(f"Could not encode frame {index}")
        images.append(image.tobytes())
    return images
